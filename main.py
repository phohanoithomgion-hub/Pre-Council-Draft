import sys
from train_sft import train_sft
from train_rl import train_rl
from pretraining import train_pretrain
from inference import run_inference

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [pretrain|sft|rl|inference] [args...]")
        return
    mode = sys.argv[1].lower()
    if mode == "pretrain":
        train_pretrain()
    elif mode == "sft":
        train_sft(sys.argv[2:])
    elif mode == "rl":
        train_rl()
    elif mode == "inference":
        run_inference(sys.argv[2:])
    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()