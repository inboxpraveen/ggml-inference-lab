"""Shared arithmetic for blog-3 (Cerebras from the outside). Everything here is computed from
results/cerebras_sources.json; nothing is typed in. Both charts3.py and build_blog3.py import it so
a chart and a sentence can never disagree."""
import os, json, math

LAB = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = json.load(open(os.path.join(LAB, 'results', 'cerebras_sources.json'), encoding='utf-8'))
W, G, M, B, P = SRC['wafer'], SRC['gpu'], SRC['models'], SRC['business'], SRC['papers']


def v(section, key):
    return SRC[section][key]['v']


# ------------------------------------------------------------------ the wafer, derived
CORES = v('wafer', 'wse3_cores')
SRAM_B = v('wafer', 'wse3_sram_gb') * 1e9
SRAM_BW = v('wafer', 'wse3_sram_bw_pbs') * 1e15
FABRIC_BW = v('wafer', 'wse3_fabric_bw_pbs') * 1e15
DENSE_FLOPS = v('wafer', 'wse3_dense_pflops') * 1e15
SPARSE_FLOPS = v('wafer', 'wse3_sparse_pflops') * 1e15
CLOCK = v('wafer', 'core_clock_ghz_wse2') * 1e9
SIMD = v('wafer', 'core_simd_wse3')
CORE_SRAM_B = v('wafer', 'core_sram_kb') * 1024
T_SRAM_BW = v('wafer', 'wse3t_sram_bw_pbs') * 1e15

# what the spec sheet implies per core
sram_from_cores_gb = CORES * CORE_SRAM_B / 1e9                    # 900,000 x 48 KB
bytes_per_core_cycle = SRAM_BW / (CORES * CLOCK)                  # 21 PB/s spread over cores and cycles
dense_from_cores_pf = CORES * SIMD * 2 * CLOCK / 1e15             # 8 FMAC = 16 flop per core per cycle
sparse_to_dense = SPARSE_FLOPS / DENSE_FLOPS
h100_bw = v('gpu', 'h100_bw_tbs') * 1e12
b200_bw = v('gpu', 'b200_bw_tbs') * 1e12
bw_ratio_h100 = SRAM_BW / h100_bw
wafer_side_cores = math.sqrt(CORES)                                # a square mesh this many cores across
wafer_cross_us = wafer_side_cores / CLOCK * 1e6                    # one hop per clock, edge to edge


# ------------------------------------------------------------------ models
def bytes_total(mk, bits=16):
    return M[mk]['params_b'] * 1e9 * bits / 8


def bytes_active(mk, bits=16):
    return M[mk]['active_b'] * 1e9 * bits / 8


def kv_per_token(mk):
    """f16 bytes of cache per token of context (K and V, or the MLA latent once)."""
    m = M[mk]
    if m.get('mla'):
        return m['layers'] * m['head_dim'] * 2
    return m['attn_layers'] * m['kv_heads'] * m['head_dim'] * 2 * 2


HEADROOM = 0.85  # share of a wafer's SRAM I assume is usable for weights (activations, cache, buffers take the rest)


def wafers_needed(mk, bits=16, headroom=HEADROOM):
    return math.ceil(bytes_total(mk, bits) / (SRAM_B * headroom))


def wafers_min(mk, bits=16):
    """no headroom at all: the hard floor"""
    return math.ceil(bytes_total(mk, bits) / SRAM_B)


# ------------------------------------------------------------------ ceilings for one 70B at 16-bit
def ceiling(bw_total, mk='llama31_70b', bits=16):
    return bw_total / bytes_active(mk, bits)


def implied_us_per_layer(tps, layers):
    return 1e6 / (tps * layers)


# ------------------------------------------------------------------ per-layer budget model
# t_layer = bytes_per_layer / (devices x bandwidth) + flops_per_layer / (devices x dense) + n_sync x t_sync
# GPU t_sync is fitted from the one GPU point that has no speculative decoding (H100 TP8, 70B, 128 tok/s).
def gpu_sync_fit():
    row = SRC['gpu_baselines_2024']['h100_70b_tps']['v']
    layers = M['llama31_70b']['layers']
    total = 1e6 / (row * layers)
    byte_term = bytes_active('llama31_70b') / layers / (8 * h100_bw) * 1e6
    return (total - byte_term) / 2, total, byte_term


GPU_T_SYNC, _, _ = gpu_sync_fit()


def wafer_sync_fit():
    """fitted from the two Cerebras points published before speculative decoding (Aug 2024)."""
    out = {}
    for r in SRC['speeds']:
        if r['who'] == 'Cerebras' and r['spec'] is False:
            out[r['model']] = implied_us_per_layer(r['tps'], r['layers'])
    return out


WAFER_SYNC = wafer_sync_fit()


def layer_budget(device, mk='llama31_70b', bits=16, n_dev=8):
    """microseconds per layer per decode step, split into bytes, compute, sync"""
    m = M[mk]
    per_layer_bytes = bytes_active(mk, bits) / m['layers']
    per_layer_flop = 2 * m['active_b'] * 1e9 / m['layers']
    if device == 'h100':
        return per_layer_bytes / (n_dev * h100_bw) * 1e6, per_layer_flop / (n_dev * 989e12) * 1e6, 2 * GPU_T_SYNC
    if device == 'b200':
        return per_layer_bytes / (n_dev * b200_bw) * 1e6, per_layer_flop / (n_dev * 2.25e15) * 1e6, 2 * GPU_T_SYNC
    # weights are stationary: one layer lives on (wafers / layers) of a wafer and only those cores serve it
    share = wafers_needed(mk, bits) / m['layers']
    total = WAFER_SYNC[m['label']] if m['label'] in WAFER_SYNC else None
    if device == 'wse3':
        b, c = per_layer_bytes / (SRAM_BW * share) * 1e6, per_layer_flop / (DENSE_FLOPS * share) * 1e6
        return b, c, total - b - c
    if device == 'wse3t':
        b, c = per_layer_bytes / (T_SRAM_BW * share) * 1e6, per_layer_flop / (2 * DENSE_FLOPS * share) * 1e6
        return b, c, (total - 2 * b - 2 * c) / 2
    raise KeyError(device)


def wafer_layer_share(mk, bits=16):
    return wafers_needed(mk, bits) / M[mk]['layers']


# ------------------------------------------------------------------ economics (assumptions stated in the post)
POWER_PRICE = 0.10          # USD per kWh
AMORT_YEARS = 4
def hourly_cost(n_wafers, capex_per_wafer):
    return n_wafers * (capex_per_wafer / (AMORT_YEARS * 8760) + v('wafer', 'wafer_power_kw_semi') * POWER_PRICE)


def users_to_break_even(price_per_m_out, tps_per_user, n_wafers, capex_per_wafer):
    rev_per_user_hour = tps_per_user * 3600 / 1e6 * price_per_m_out
    return hourly_cost(n_wafers, capex_per_wafer) / rev_per_user_hour


if __name__ == '__main__':
    print(f'SRAM from cores: {sram_from_cores_gb:.1f} GB; bytes/core/cycle: {bytes_per_core_cycle:.1f}; dense from cores: {dense_from_cores_pf:.1f} PF; sparse:dense {sparse_to_dense:.0f}')
    print(f'bandwidth vs H100: {bw_ratio_h100:.0f}x; wafer side ~{wafer_side_cores:.0f} cores; edge-to-edge {wafer_cross_us:.2f} us')
    for mk in M:
        print(f"{M[mk]['label']:>18}: {bytes_total(mk)/1e9:6.0f} GB @16b, wafers 16/8/4-bit = {wafers_needed(mk,16)}/{wafers_needed(mk,8)}/{wafers_needed(mk,4)} (floor {wafers_min(mk,16)}), KV {kv_per_token(mk)/1024:.0f} KiB/token")
    print('GPU sync fit', GPU_T_SYNC, 'wafer sync', WAFER_SYNC)
    for d in ['h100', 'b200', 'wse3', 'wse3t']:
        print(d, ['%.2f' % x for x in layer_budget(d)])
    for r in SRC['speeds']:
        if r['layers']:
            print(f"{r['model']:>24} {r['who']:>22} {r['tps']:>5} -> {implied_us_per_layer(r['tps'], r['layers']):5.1f} us/layer/token spec={r['spec']}")
    print('break-even users 70B, $0.60, BOM:', users_to_break_even(0.60, 2100, 4, v('wafer', 'bom_per_wafer_rack_usd')), 'list:', users_to_break_even(0.60, 2100, 4, v('wafer', 'list_price_per_system_usd')))
