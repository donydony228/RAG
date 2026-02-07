#!/usr/bin/env python3
# 將 credentials 文件編碼為 base64 格式
# 用於在 Railway 環境變數中設定
import base64
import sys
from pathlib import Path

def encode_file(file_path: str) -> str:
    """將文件內容編碼為 base64"""
    path = Path(file_path)
    if not path.exists():
        print(f"錯誤: 文件不存在 {file_path}")
        sys.exit(1)
    
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode('utf-8')
    return encoded

def main():
    """主程式"""
    print("Gmail Credentials Base64 編碼工具")
    print("=" * 50)
    
    multi_account = input("是否使用多帳號模式？(y/n): ").lower() == 'y'
    
    if multi_account:
        print("\n多帳號模式")
        for i in range(1, 4):
            print(f"\n--- 帳號 {i} ---")
            
            cred_path = f"credentials/credentials_account{i}.json"
            token_path = f"credentials/token_account{i}.json"
            
            if Path(cred_path).exists():
                encoded_cred = encode_file(cred_path)
                print(f"GMAIL_CREDENTIALS_ACCOUNT{i}_BASE64=")
                print(encoded_cred)
                print()
            
            if Path(token_path).exists():
                encoded_token = encode_file(token_path)
                print(f"GMAIL_TOKEN_ACCOUNT{i}_BASE64=")
                print(encoded_token)
                print()
    else:
        print("\n單一帳號模式")
        
        cred_path = "credentials/credentials.json"
        token_path = "credentials/token.json"
        
        if Path(cred_path).exists():
            encoded_cred = encode_file(cred_path)
            print("GMAIL_CREDENTIALS_BASE64=")
            print(encoded_cred)
            print()
        
        if Path(token_path).exists():
            encoded_token = encode_file(token_path)
            print("GMAIL_TOKEN_BASE64=")
            print(encoded_token)
            print()
    
    print("\n請將以上環境變數複製到 Railway 環境變數設定中")

if __name__ == "__main__":
    main()
