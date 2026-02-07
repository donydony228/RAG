#!/usr/bin/env python3
"""
為多個 Gmail 帳號分別授權
"""

from services.gmail_service import authenticate

def main():
    accounts = [
        {
            "number": 1,
            "label": "個人",
            "credentials": "credentials/credentials_account1.json",
            "token": "credentials/token_account1.json"
        },
        {
            "number": 2,
            "label": "工作",
            "credentials": "credentials/credentials_account2.json",
            "token": "credentials/token_account2.json"
        },
        {
            "number": 3,
            "label": "紐約大學",
            "credentials": "credentials/credentials_account3.json",
            "token": "credentials/token_account3.json"
        }
    ]
    
    print("=" * 70)
    print("Gmail 多帳號授權")
    print("=" * 70)
    print()
    
    for account in accounts:
        print(f"\n{'='*70}")
        print(f"帳號 {account['number']}: {account['label']}")
        print(f"{'='*70}")
        print()
        print(f"即將開啟瀏覽器進行授權")
        print(f"請登入你的 **{account['label']}** Gmail 帳號")
        print()
        
        input("按 Enter 繼續...")
        
        try:
            authenticate(
                credentials_path=account['credentials'],
                token_path=account['token']
            )
            print(f"✓ {account['label']} 授權成功！")
        except Exception as e:
            print(f"✗ {account['label']} 授權失敗: {e}")
    
    print("\n" + "="*70)
    print("所有帳號授權完成！")
    print("="*70)
    print("\n接下來請執行:")
    print("  python encode_credentials.py")
    print("\n然後更新 GitHub Secrets")

if __name__ == "__main__":
    main()
