#!/usr/bin/env bash
set -e

DATA_DIR="data"
mkdir -p "${DATA_DIR}"

download_yearpred() {
    echo "Downloading YearPredictionMSD..."
    wget -O "${DATA_DIR}/yearpredictionmsd.zip" \
        https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip

    echo "Unzipping YearPredictionMSD..."
    unzip -o "${DATA_DIR}/yearpredictionmsd.zip" -d "${DATA_DIR}"

    rm "${DATA_DIR}/yearpredictionmsd.zip"
}

download_higgs() {
    echo "Downloading HIGGS..."
    wget -O "${DATA_DIR}/higgs.zip" \
        https://archive.ics.uci.edu/static/public/280/higgs.zip
}

usage() {
    echo "Usage: $0 [yearpred|higgs|all]"
    echo ""
    echo "Options:"
    echo "  yearpred   Download only YearPredictionMSD"
    echo "  higgs      Download only HIGGS"
    echo "  all        Download both datasets"
    exit 1
}

# -------- Argument handling --------

if [[ $# -ne 1 ]]; then
    usage
fi

case "$1" in
    yearpred)
        download_yearpred
        ;;
    higgs)
        download_higgs
        ;;
    all)
        download_yearpred
        download_higgs
        ;;
    *)
        usage
        ;;
esac

echo "Done."
