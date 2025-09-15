#!/usr/bin/env bash
set -euo pipefail


###### WRITTEN USING CHATGPT-5 TO LOOP OVER PARAMETERS ######
# ---------------------------------------------------------
# Batch runner per sweep.py (Ultimatum networks) + rete empirica (Karate Club)
# Tempi:
#   - natural_selection: 1 100 1000 10000 20000 100000
#   - social_penalty:    1 100 1000 10000 100000
# Requisiti: python (numpy, networkx, matplotlib) + curl o wget
# Variabili (sovrascrivibili via env):
#   PYTHON, SCRIPT, OUTBASE, SEED, N, AVGK, M, BINS
#   TIMEPOINTS_NS, TIMEPOINTS_SP, HEATMAP_TIMES_NS, HEATMAP_TIMES_SP, MIXPROBS
#   SWEEP_N_LIST, SWEEP_AVGK_LIST, SWEEP_BAM_LIST, SWEEP_STEPS, SWEEP_TAILFRAC
# ---------------------------------------------------------

PYTHON_BIN="${PYTHON:-python3}"
SCRIPT="${SCRIPT:-sweep.py}"

SEED="${SEED:-42}"
N="${N:-500}"
AVGK="${AVGK:-6}"          # ER: grado medio
M="${M:-3}"                # BA: m ~ AVGK/2
BINS="${BINS:-40}"

TIMEPOINTS_NS="${TIMEPOINTS_NS:-1 100 1000 10000 20000 100000}"
TIMEPOINTS_SP="${TIMEPOINTS_SP:-1 100 1000 10000 100000}"

HEATMAP_TIMES_NS="${HEATMAP_TIMES_NS:-100000}"
HEATMAP_TIMES_SP="${HEATMAP_TIMES_SP:-100000}"

MIXPROBS="${MIXPROBS:-0.34 0.33 0.33}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTBASE="${OUTBASE:-outputs_batch_${STAMP}}"
mkdir -p "${OUTBASE}"

echo "[INFO] python:   ${PYTHON_BIN}"
echo "[INFO] script:   ${SCRIPT}"
echo "[INFO] outbase:  ${OUTBASE}"
echo "[INFO] n=${N}, <k>_ER=${AVGK}, m_BA=${M}, bins=${BINS}"
echo "[INFO] times(NS): ${TIMEPOINTS_NS}"
echo "[INFO] times(SP): ${TIMEPOINTS_SP}"
echo

# ----------------- Edgelist EMP (Karate Club) -----------------
DATA_DIR="${DATA_DIR:-data}"
mkdir -p "${DATA_DIR}"
EDGELIST_PATH="${EDGELIST_PATH:-${DATA_DIR}/karate.edgelist}"
URL_RAW="https://raw.githubusercontent.com/freditation/karate-club/master/karate.edgelist"

# Scarica se il file non esiste o è vuoto
if [[ ! -s "${EDGELIST_PATH}" ]]; then
  echo "[DL] Scarico Karate Club edgelist in ${EDGELIST_PATH}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail "$URL_RAW" -o "${EDGELIST_PATH}" || true
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${EDGELIST_PATH}" "$URL_RAW" || true
  else
    echo "[WARN] Né curl né wget disponibili: proseguo senza download."
  fi
fi

# Se l’utente ha un CSV con header/virgole (fromPersonId,toPersonId), converti in space-separated
EDGELIST_WS="${DATA_DIR}/karate_ws.edgelist"
if [[ -s "${EDGELIST_PATH}" ]] && head -n1 "${EDGELIST_PATH}" | grep -q "," ; then
  echo "[INFO] Rilevato CSV con virgole. Converto in ${EDGELIST_WS}"
  if head -n1 "${EDGELIST_PATH}" | grep -qi '[a-z]'; then
    tail -n +2 "${EDGELIST_PATH}" | awk -F, '{print $1, $2}' > "${EDGELIST_WS}"
  else
    awk -F, '{print $1, $2}' "${EDGELIST_PATH}" > "${EDGELIST_WS}"
  fi
  EDGELIST_PATH="${EDGELIST_WS}"
fi

if [[ -s "${EDGELIST_PATH}" ]]; then
  echo "[OK] Edgelist EMP: ${EDGELIST_PATH} ($(wc -l < "${EDGELIST_PATH}") righe)"
else
  echo "[WARN] Nessuna edgelist valida trovata; le corse EMP useranno il fallback del loader Python (karate builtin, se implementato)."
fi

# -------- helper: una corsa singola (plot multi-tempo, heatmap, <p>_k/<q>_k) ------
run_case () {
  local TOPO="$1"    # ER | BA | EMP
  local RULE="$2"    # natural_selection | social_penalty
  local MODE="$3"    # A | B | C | MIX

  local SUB="${OUTBASE}/plots/${TOPO}/${RULE}/${MODE}"
  mkdir -p "${SUB}"
  echo "[RUN] ${TOPO}  ${RULE}  ${MODE}  ->  ${SUB}"

  local TP="${TIMEPOINTS_NS}"; local HMT="${HEATMAP_TIMES_NS}"
  if [[ "${RULE}" == "social_penalty" ]]; then
    TP="${TIMEPOINTS_SP}"; HMT="${HEATMAP_TIMES_SP}"
  fi

  local COMMON=( --topology "${TOPO}" --rule "${RULE}" -n "${N}" --avg-k "${AVGK}" --ba-m "${M}" \
                 --seed "${SEED}" --times ${TP} --bins "${BINS}" --outdir "${SUB}" \
                 --plot-q --heatmap-times ${HMT} --plot-pk --plot-qk )

  if [[ "${TOPO}" == "EMP" ]]; then
    COMMON+=( --edgelist "${EDGELIST_PATH}" )
  fi

  if [[ "${MODE}" == "MIX" ]]; then
    ${PYTHON_BIN} "${SCRIPT}" "${COMMON[@]}" --mixed --mix-probs ${MIXPROBS}
  else
    ${PYTHON_BIN} "${SCRIPT}" "${COMMON[@]}" --type "${MODE}"
  fi
}

# ---------------------- 1) tutte le combinazioni -------------------
PLAYERS=("A" "B" "C" "MIX")
TOPOLOGIES=("ER" "BA" "EMP")
RULES=("natural_selection" "social_penalty")

for topo in "${TOPOLOGIES[@]}"; do
  for rule in "${RULES[@]}"; do
    for mode in "${PLAYERS[@]}"; do
      run_case "${topo}" "${rule}" "${mode}"
    done
  done
done

# ---------------------- 2) parameter sweep (CSV, ER vs BA + EMP) ------------------
SWEEP_N_LIST="${SWEEP_N_LIST:-300 500 1000}"
SWEEP_AVGK_LIST="${SWEEP_AVGK_LIST:-4 6 8}"   # ER
SWEEP_BAM_LIST="${SWEEP_BAM_LIST:-2 3 4}"     # BA
SWEEP_STEPS="${SWEEP_STEPS:-20000}"
SWEEP_TAILFRAC="${SWEEP_TAILFRAC:-0.1}"

sweep_case () {
  local RULE="$1"    # natural_selection | social_penalty
  local MODE="$2"    # A | B | C | MIX
  local SUB="${OUTBASE}/sweep/${RULE}/${MODE}"
  mkdir -p "${SUB}"
  echo
  echo "[SWEEP] rule=${RULE}  mode=${MODE}  ->  ${SUB}"

  if [[ "${MODE}" == "MIX" ]]; then
    ${PYTHON_BIN} "${SCRIPT}" --sweep \
      --sweep-n ${SWEEP_N_LIST} \
      --sweep-avgk ${SWEEP_AVGK_LIST} \
      --sweep-bam ${SWEEP_BAM_LIST} \
      --sweep-steps "${SWEEP_STEPS}" \
      --sweep-tailfrac "${SWEEP_TAILFRAC}" \
      --rule "${RULE}" --mixed --mix-probs ${MIXPROBS} \
      --edgelist "${EDGELIST_PATH}" \
      --outdir "${SUB}"
  else
    ${PYTHON_BIN} "${SCRIPT}" --sweep \
      --sweep-n ${SWEEP_N_LIST} \
      --sweep-avgk ${SWEEP_AVGK_LIST} \
      --sweep-bam ${SWEEP_BAM_LIST} \
      --sweep-steps "${SWEEP_STEPS}" \
      --sweep-tailfrac "${SWEEP_TAILFRAC}" \
      --rule "${RULE}" --type "${MODE}" \
      --edgelist "${EDGELIST_PATH}" \
      --outdir "${SUB}"
  fi
}

for rule in "${RULES[@]}"; do
  for mode in "${PLAYERS[@]}"; do
    sweep_case "${rule}" "${mode}"
  done
done

echo
echo "[DONE] Tutto completato. Output in: ${OUTBASE}"
