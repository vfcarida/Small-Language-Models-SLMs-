import argparse
import logging
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# ==============================================================================
# ACTIVATION-AWARE WEIGHT QUANTIZATION (AWQ) CLI
# ==============================================================================
# This script automates the AWQ process for Small Language Models (SLMs).
# AWQ locks down memory bandwidth bottlenecks by converting FP16 weights into 
# INT4 arrays, but crucially preserves the ~1% of "salient" weights to prevent 
# numerical collapse (unlike SmoothQuant).
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """
    Sets up the Command Line Interface (CLI) parsing rules.
    """
    parser = argparse.ArgumentParser(description="AutoAWQ Quantization CLI for SLMs.")
    parser.add_argument(
        "--model_id", 
        type=str, 
        required=True, 
        help="Hugging Face Model ID (e.g., Qwen/Qwen2.5-1.5B)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./quantized_model", 
        help="Destination directory for the quantized INT4 model"
    )
    
    # --------------------------------------------------------------------------
    # ARCHITECTURAL CONSTRAINTS
    # We hardcode `w_bit` and `q_group_size` because 4-bit and 128 block size 
    # are mathematically proven to be the geometric optimal limit for SLMs.
    # Going to INT3 destroys coherence; Group size 64 adds too much metadata overhead.
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--w_bit", 
        type=int, 
        default=4, 
        choices=[4],
        help="Locked to 4-bits (Design Constraint for SLMs)"
    )
    parser.add_argument(
        "--q_group_size", 
        type=int, 
        default=128, 
        choices=[128],
        help="Scale block group size (Design Constraint)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Initiating AWQ Process for model: {args.model_id}")
    logger.info(f"Static Configuration: w_bit={args.w_bit}, q_group_size={args.q_group_size}")

    # --------------------------------------------------------------------------
    # 1. DEFINE QUANTIZATION CONFIGURATION
    # --------------------------------------------------------------------------
    quant_config = {
        # zero_point allows asymmetric quantization, capturing negative/positive skews better
        "zero_point": True, 
        # Number of parameters sharing a single scaling factor
        "q_group_size": args.q_group_size, 
        # Target bit depth
        "w_bit": args.w_bit, 
        # Target General Matrix Multiply optimizations
        "version": "GEMM" 
    }
    
    try:
        # --------------------------------------------------------------------------
        # 2. LOAD ORIGINAL MODEL
        # --------------------------------------------------------------------------
        logger.info("Loading original massive FP16/BF16 model into VRAM...")
        # AutoAWQ bypasses standard Transformers loaders to inject custom CUDA kernels
        model = AutoAWQForCausalLM.from_pretrained(args.model_id, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        
        # --------------------------------------------------------------------------
        # 3. EXECUTE ACTIVATION-AWARE QUANTIZATION
        # --------------------------------------------------------------------------
        # AutoAWQ runs a subset of WikiText through the model to identify the 1% 
        # of salient channels (the massive activation outliers).
        # It protects those channels via scaling, then crushes the remaining 99% to INT4.
        logger.info("Executing Activation-Aware Math and quantizing to INT4...")
        model.quantize(tokenizer, quant_config=quant_config)
        
        # --------------------------------------------------------------------------
        # 4. SAVE EXPORTED MODEL
        # --------------------------------------------------------------------------
        # The output model is now drastically smaller (e.g., 1.5B params = ~1GB on disk)
        logger.info(f"Saving quantized model to: {args.output_dir}")
        model.save_quantized(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logger.info("AWQ Compilation finished successfully. Ready for Edge Deployment.")
        
    except Exception as e:
        logger.error(f"Catastrophic failure during AWQ compilation: {e}")
        # We explicitly raise the error so CI/CD pipelines fail immediately
        raise RuntimeError(f"AWQ Compilation Failed: {e}")

if __name__ == "__main__":
    main()
