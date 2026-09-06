"""
Master Pipeline Script: Hierarchical Hate Speech Detection for Tanglish
=======================================================================
Description:
    Sequentially executes Phase 1 through Phase 5 in order.
    Catches and logs any phase failure, halting execution safely if an error occurs.

Outputs:
    - All phase joblib artifacts, PyTorch saved model weights (.pt),
      confusion matrix plot, and attention visualization images.
"""

import sys

def main():
    print("=" * 80)
    print(" 🚀 STARTING FULL HIERARCHICAL HATE SPEECH DETECTION PIPELINE FOR TANGLISH")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Phase 1: Environment Setup & Data Download
    # --------------------------------------------------------------------------
    try:
        print("\n▶ Running Phase 1: Setup & Data Download...")
        import phase1_setup
        phase1_setup.install_packages()
        phase1_setup.check_gpu()
        extract_dir = phase1_setup.download_dataset()
        phase1_setup.load_and_clean_data(extract_dir)
        print("✅ Phase 1 complete — data saved to phase1_data.joblib")
    except Exception as e:
        print(f"❌ Phase 1 FAILED with error: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Phase 2: Preprocessing & Tanglish Slang Normalization
    # --------------------------------------------------------------------------
    try:
        print("\n▶ Running Phase 2: Preprocessing & DataLoaders...")
        import phase2_preprocessing
        phase2_preprocessing.main()
        print("✅ Phase 2 complete — preprocessed data saved to phase2_data.joblib")
    except Exception as e:
        print(f"❌ Phase 2 FAILED with error: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Phase 3: Sentence-Only Baselines (XLM-R & MuRIL)
    # --------------------------------------------------------------------------
    try:
        print("\n▶ Running Phase 3: Sentence-Only Baselines (XLM-R & MuRIL)...")
        import phase3_baselines
        phase3_baselines.main()
        print("✅ Phase 3 complete — baseline results saved to phase3_results.joblib")
    except Exception as e:
        print(f"❌ Phase 3 FAILED with error: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Phase 4: Proposed Hierarchical Fusion Model & Ablation Study
    # --------------------------------------------------------------------------
    try:
        print("\n▶ Running Phase 4: Proposed Hierarchical Model & Ablation Study...")
        import phase4_hierarchical
        phase4_hierarchical.main()
        print("✅ Phase 4 complete — hierarchical model results saved to phase4_results.joblib")
    except Exception as e:
        print(f"❌ Phase 4 FAILED with error: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Phase 5: Evaluation, Visualizations & Error Analysis
    # --------------------------------------------------------------------------
    try:
        print("\n▶ Running Phase 5: Comprehensive Evaluation & Attention Visualizations...")
        import phase5_evaluation
        phase5_evaluation.main()
        print("✅ Phase 5 complete — visualizations saved to confusion_matrix.png & attention_viz_ex1.png")
    except Exception as e:
        print(f"❌ Phase 5 FAILED with error: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 All phases complete. Check confusion_matrix.png and attention_viz_ex1.png for outputs. Your results are in phase4_results.joblib")
    print("=" * 80)

if __name__ == "__main__":
    main()
