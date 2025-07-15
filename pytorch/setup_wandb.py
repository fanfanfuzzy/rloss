#!/usr/bin/env python3
"""
Secure Weights & Biases setup script for rloss training
Handles API key input without storing it in the repository
"""
import os
import getpass
import sys

def setup_wandb_credentials():
    """Securely set up W&B credentials"""
    print("🔐 Weights & Biases Setup")
    print("=" * 50)
    
    if os.environ.get('WANDB_API_KEY'):
        print("✅ WANDB_API_KEY is already set in environment")
        return True
    
    print("To use Weights & Biases for experiment tracking:")
    print("1. Sign up at https://wandb.ai/ (free account available)")
    print("2. Get your API key from https://wandb.ai/authorize")
    print("3. Enter it below (input will be hidden)")
    print()
    
    try:
        api_key = getpass.getpass("Enter your W&B API key: ").strip()
        
        if not api_key:
            print("❌ No API key provided")
            return False
        
        if len(api_key) < 20:
            print("❌ API key seems too short. Please check and try again.")
            return False
        
        os.environ['WANDB_API_KEY'] = api_key
        
        env_file = '/workspace/.env'
        try:
            with open(env_file, 'a') as f:
                f.write(f"\nexport WANDB_API_KEY={api_key}\n")
            print("✅ W&B API key set successfully!")
            print("✅ API key saved to /workspace/.env for persistence")
            print("💡 To activate in current shell, run: source /workspace/.env")
        except Exception as e:
            print("✅ W&B API key set for current session!")
            print("💡 To make this permanent, add to your shell profile:")
            print(f"   export WANDB_API_KEY={api_key}")
        print()
        print("🚀 You can now run training with --use-wandb flag")
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False

def check_wandb_status():
    """Check current W&B setup status"""
    try:
        import wandb
        print("✅ wandb package is installed")
        
        if os.environ.get('WANDB_API_KEY'):
            print("✅ WANDB_API_KEY is set")
            print("🎯 Ready for W&B logging!")
            return True
        else:
            print("⚠️  WANDB_API_KEY not set")
            return False
            
    except ImportError:
        print("❌ wandb package not installed")
        print("   Install with: pip install wandb")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_wandb_status()
    else:
        setup_wandb_credentials()
