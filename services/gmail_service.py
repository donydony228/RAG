"""
Gmail API 服務
處理 Gmail API 連接、郵件獲取等功能
"""

import os
import base64
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API 權限範圍
# 如果修改這些範圍，需要刪除 token.json 重新授權
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',  # 讀取郵件
    # 'https://www.googleapis.com/auth/gmail.send',    # 發送郵件（如果需要）
]


def authenticate(credentials_path: str = 'credentials.json',
                token_path: str = 'token.json',
                credentials_base64_env: str = None,
                token_base64_env: str = None) -> Credentials:
    """
    Gmail API 認證

    優先從環境變量讀取 base64 編碼的 credentials/token
    如果環境變量不存在，則從文件讀取

    Args:
        credentials_path: OAuth 2.0 憑證檔案路徑
        token_path: Token 儲存路徑
        credentials_base64_env: Credentials base64 環境變量名稱
        token_base64_env: Token base64 環境變量名稱

    Returns:
        Credentials: 認證憑證
    """
    creds = None

    # 優先從環境變量讀取 token
    if token_base64_env and os.getenv(token_base64_env):
        try:
            print(f"從環境變量 {token_base64_env} 讀取 token...")
            token_data = base64.b64decode(os.getenv(token_base64_env))
            creds = pickle.loads(token_data)
            print("✓ Token 從環境變量加載成功")
        except Exception as e:
            print(f"從環境變量讀取 token 失敗: {e}")
            creds = None
    # 否則從文件讀取
    elif os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
        print(f"✓ Token 從文件 {token_path} 加載成功")

    # 如果沒有有效憑證，進行授權流程
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Token 過期但可以更新
            print("Token 已過期，正在更新...")
            creds.refresh(Request())
        else:
            # 需要重新授權 - 從環境變量或文件讀取 credentials
            print("需要重新授權...")

            # 優先從環境變量讀取 credentials
            if credentials_base64_env and os.getenv(credentials_base64_env):
                try:
                    print(f"從環境變量 {credentials_base64_env} 讀取 credentials...")
                    import json
                    import tempfile
                    cred_data = base64.b64decode(os.getenv(credentials_base64_env))
                    cred_json = json.loads(cred_data)

                    # 創建臨時文件供 InstalledAppFlow 使用
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(cred_json, f)
                        temp_cred_path = f.name

                    flow = InstalledAppFlow.from_client_secrets_file(temp_cred_path, SCOPES)
                    creds = flow.run_local_server(port=0)

                    os.unlink(temp_cred_path)  # 刪除臨時文件
                    print("✓ 從環境變量授權成功")
                except Exception as e:
                    raise Exception(f"從環境變量授權失敗: {e}")
            # 否則從文件讀取
            elif os.path.exists(credentials_path):
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                print(f"✓ 從文件 {credentials_path} 授權成功")
            else:
                raise FileNotFoundError(
                    f"找不到憑證檔案: {credentials_path}\n"
                    f"也沒有設置環境變量: {credentials_base64_env}\n"
                    f"請從 Google Cloud Console 下載 OAuth 2.0 憑證"
                )

        # 儲存 token 供下次使用（僅當使用文件模式時）
        if not token_base64_env and token_path:
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
            print(f"Token 已保存到 {token_path}")

    return creds


def get_gmail_service(credentials_path: str = 'credentials.json',
                     token_path: str = 'token.json',
                     credentials_base64_env: str = None,
                     token_base64_env: str = None):
    """
    建立 Gmail API 服務實例

    Args:
        credentials_path: OAuth 2.0 憑證檔案路徑
        token_path: Token 儲存路徑
        credentials_base64_env: Credentials base64 環境變量名稱
        token_base64_env: Token base64 環境變量名稱

    Returns:
        Gmail API 服務實例
    """
    creds = authenticate(credentials_path, token_path, credentials_base64_env, token_base64_env)

    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f'建立 Gmail 服務失敗: {error}')
        raise


def parse_time_range(time_range: str) -> datetime:
    """
    解析時間範圍字串，回傳開始時間

    Args:
        time_range: 時間範圍 ("24h", "7d", "30d" 等)

    Returns:
        datetime: 開始時間
    """
    now = datetime.now()

    if time_range.endswith('h'):
        # 小時
        hours = int(time_range[:-1])
        return now - timedelta(hours=hours)
    elif time_range.endswith('d'):
        # 天
        days = int(time_range[:-1])
        return now - timedelta(days=days)
    elif time_range.endswith('w'):
        # 週
        weeks = int(time_range[:-1])
        return now - timedelta(weeks=weeks)
    else:
        # 預設 24 小時
        return now - timedelta(hours=24)


def decode_message_part(part: Dict) -> str:
    """
    解碼郵件內容

    Args:
        part: 郵件部分

    Returns:
        str: 解碼後的內容
    """
    if 'data' in part['body']:
        data = part['body']['data']
        # Base64 解碼
        decoded_bytes = base64.urlsafe_b64decode(data)
        return decoded_bytes.decode('utf-8', errors='ignore')
    return ''


def get_message_body(payload: Dict) -> str:
    """
    提取郵件正文

    Args:
        payload: 郵件 payload

    Returns:
        str: 郵件正文
    """
    body = ''

    if 'parts' in payload:
        # 多部分郵件
        for part in payload['parts']:
            mime_type = part.get('mimeType', '')

            # 優先提取純文字
            if mime_type == 'text/plain':
                body += decode_message_part(part)
            elif mime_type == 'text/html' and not body:
                # 如果沒有純文字，使用 HTML
                body += decode_message_part(part)
            elif 'parts' in part:
                # 遞迴處理嵌套部分
                body += get_message_body(part)
    else:
        # 單一部分郵件
        body = decode_message_part(payload)

    return body.strip()


def get_header_value(headers: List[Dict], name: str) -> str:
    """
    從郵件標頭中提取指定欄位的值

    Args:
        headers: 郵件標頭列表
        name: 欄位名稱

    Returns:
        str: 欄位值
    """
    for header in headers:
        if header['name'].lower() == name.lower():
            return header['value']
    return ''


def fetch_emails(service,
                time_range: str = '24h',
                max_emails: int = 50,
                query: str = '') -> List[Dict]:
    """
    獲取郵件

    Args:
        service: Gmail API 服務實例
        time_range: 時間範圍 ("24h", "7d", "30d" 等)
        max_emails: 最多獲取郵件數
        query: Gmail 搜尋查詢 (例如: "is:unread", "from:example@gmail.com")

    Returns:
        List[Dict]: 郵件列表
    """
    try:
        # 計算時間範圍
        start_time = parse_time_range(time_range)
        after_timestamp = int(start_time.timestamp())

        # 建立搜尋查詢
        search_query = f'after:{after_timestamp}'
        if query:
            search_query += f' {query}'

        print(f"搜尋郵件: {search_query}")

        # 獲取郵件 ID 列表
        results = service.users().messages().list(
            userId='me',
            q=search_query,
            maxResults=max_emails
        ).execute()

        messages = results.get('messages', [])

        if not messages:
            print('沒有找到郵件')
            return []

        print(f'找到 {len(messages)} 封郵件，開始獲取詳細內容...')

        # 獲取每封郵件的詳細內容
        emails = []
        for i, message in enumerate(messages, 1):
            try:
                msg = service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()

                # 提取郵件資訊
                headers = msg['payload']['headers']

                email_data = {
                    'id': msg['id'],
                    'thread_id': msg['threadId'],
                    'subject': get_header_value(headers, 'Subject'),
                    'from': get_header_value(headers, 'From'),
                    'to': get_header_value(headers, 'To'),
                    'date': get_header_value(headers, 'Date'),
                    'body': get_message_body(msg['payload']),
                    'snippet': msg.get('snippet', ''),
                    'labels': msg.get('labelIds', []),
                }

                emails.append(email_data)

                print(f'[{i}/{len(messages)}] {email_data["subject"][:50]}...')

            except HttpError as error:
                print(f'獲取郵件失敗 {message["id"]}: {error}')
                continue

        print(f'成功獲取 {len(emails)} 封郵件')
        return emails

    except HttpError as error:
        print(f'搜尋郵件失敗: {error}')
        return []


def fetch_emails_from_gmail(time_range: str = '24h',
                            max_emails: int = 50,
                            query: str = '',
                            credentials_path: str = 'credentials.json',
                            token_path: str = 'token.json',
                            account_label: Optional[str] = None,
                            credentials_base64_env: str = None,
                            token_base64_env: str = None) -> List[Dict]:
    """
    從 Gmail 獲取郵件（完整流程）

    Args:
        time_range: 時間範圍 ("24h", "7d", "30d" 等)
        max_emails: 最多獲取郵件數
        query: Gmail 搜尋查詢
        credentials_path: OAuth 2.0 憑證檔案路徑
        token_path: Token 儲存路徑
        account_label: 帳號標籤（用於多帳號場景）
        credentials_base64_env: Credentials base64 環境變量名稱
        token_base64_env: Token base64 環境變量名稱

    Returns:
        List[Dict]: 郵件列表
    """
    # 只在使用預設值時才從環境變數讀取路徑
    # 這樣多帳號模式下傳入的特定路徑不會被覆蓋
    if credentials_path == 'credentials.json':
        credentials_path = os.getenv('GMAIL_CREDENTIALS_PATH', credentials_path)
    if token_path == 'token.json':
        token_path = os.getenv('GMAIL_TOKEN_PATH', token_path)

    # 建立服務
    service = get_gmail_service(credentials_path, token_path, credentials_base64_env, token_base64_env)

    # 獲取郵件
    emails = fetch_emails(service, time_range, max_emails, query)

    # 如果有指定帳號標籤，添加到每封郵件中
    if account_label:
        for email in emails:
            email['account'] = account_label

    return emails


def fetch_emails_from_multiple_accounts(
    accounts: List[Dict[str, str]],
    time_range: str = '24h',
    max_emails_per_account: int = 50,
    query: str = ''
) -> List[Dict]:
    """
    從多個 Gmail 帳號獲取郵件

    Args:
        accounts: 帳號配置列表，每個元素包含:
            {
                'label': '帳號標籤（例如：個人、工作、其他）',
                'credentials_path': 'credentials 檔案路徑',
                'token_path': 'token 檔案路徑'
            }
        time_range: 時間範圍 ("24h", "7d", "30d" 等)
        max_emails_per_account: 每個帳號最多獲取郵件數
        query: Gmail 搜尋查詢（套用到所有帳號）

    Returns:
        List[Dict]: 所有帳號的郵件列表（合併）

    Example:
        accounts = [
            {
                'label': '個人',
                'credentials_path': 'credentials_account1.json',
                'token_path': 'token_account1.json'
            },
            {
                'label': '工作',
                'credentials_path': 'credentials_account2.json',
                'token_path': 'token_account2.json'
            },
            {
                'label': '其他',
                'credentials_path': 'credentials_account3.json',
                'token_path': 'token_account3.json'
            }
        ]
        emails = fetch_emails_from_multiple_accounts(accounts, time_range='7d')
    """
    all_emails = []

    print(f"\n{'=' * 60}")
    print(f"從 {len(accounts)} 個帳號獲取郵件")
    print(f"{'=' * 60}\n")

    for i, account in enumerate(accounts, 1):
        label = account.get('label', f'Account {i}')
        credentials_path = account.get('credentials_path')
        token_path = account.get('token_path')
        credentials_base64_env = account.get('credentials_base64_env')
        token_base64_env = account.get('token_base64_env')

        print(f"[{i}/{len(accounts)}] 正在獲取帳號: {label}")
        if credentials_base64_env and token_base64_env:
            print(f"   使用環境變量: {credentials_base64_env}, {token_base64_env}")
        else:
            print(f"   使用文件: {credentials_path}, {token_path}")

        try:
            # 獲取該帳號的郵件
            emails = fetch_emails_from_gmail(
                time_range=time_range,
                max_emails=max_emails_per_account,
                query=query,
                credentials_path=credentials_path,
                token_path=token_path,
                account_label=label,
                credentials_base64_env=credentials_base64_env,
                token_base64_env=token_base64_env
            )

            all_emails.extend(emails)
            print(f"✓ 帳號 [{label}] 獲取成功: {len(emails)} 封郵件\n")

        except Exception as e:
            print(f"✗ 帳號 [{label}] 獲取失敗: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    print(f"{'=' * 60}")
    print(f"總共獲取: {len(all_emails)} 封郵件")
    print(f"{'=' * 60}\n")

    return all_emails


# 測試用主程式
if __name__ == '__main__':
    """
    測試 Gmail API

    使用方式:

    # 測試單一帳號
    python services/gmail_service.py

    # 測試多個帳號（需要先設定好各帳號的憑證）
    python services/gmail_service.py --multi
    """
    import sys

    # 檢查是否要測試多帳號
    test_multi_accounts = '--multi' in sys.argv

    if test_multi_accounts:
        print("\n" + "=" * 60)
        print("多帳號 Gmail API 測試")
        print("=" * 60 + "\n")

        # 定義多個帳號配置
        accounts = [
            {
                'label': '個人信箱',
                'credentials_path': 'credentials_account1.json',
                'token_path': 'token_account1.json'
            },
            {
                'label': '工作信箱',
                'credentials_path': 'credentials_account2.json',
                'token_path': 'token_account2.json'
            },
            {
                'label': '其他信箱',
                'credentials_path': 'credentials_account3.json',
                'token_path': 'token_account3.json'
            }
        ]

        try:
            # 從所有帳號獲取郵件
            all_emails = fetch_emails_from_multiple_accounts(
                accounts=accounts,
                time_range='24h',
                max_emails_per_account=20,
                query=''
            )

            # 顯示郵件摘要（按帳號分組）
            print("\n" + "=" * 60)
            print("郵件摘要（按帳號分組）:")
            print("=" * 60)

            # 按帳號分組顯示
            for account_config in accounts:
                label = account_config['label']
                account_emails = [e for e in all_emails if e.get('account') == label]

                if account_emails:
                    print(f"\n{label} ({len(account_emails)} 封)")
                    print("-" * 60)
                    for i, email in enumerate(account_emails[:5], 1):  # 只顯示前 5 封
                        print(f"  [{i}] {email['subject'][:50]}")
                        print(f"      寄件者: {email['from'][:40]}")
                        print(f"      日期: {email['date']}")

            print("\n" + "=" * 60)
            print(f"總計: {len(all_emails)} 封郵件")
            print("=" * 60)

        except Exception as e:
            print(f"測試失敗: {e}")

    else:
        print("\n" + "=" * 60)
        print("單一帳號 Gmail API 測試")
        print("=" * 60 + "\n")

        try:
            emails = fetch_emails_from_gmail(
                time_range='24h',
                max_emails=20,
                query='',
            )

            print("\n" + "=" * 60)
            print("郵件摘要:")
            print("=" * 60)

            for i, email in enumerate(emails, 1):
                print(f"\n[{i}] {email['subject']}")
                print(f"    寄件者: {email['from']}")
                print(f"    日期: {email['date']}")
                print(f"    預覽: {email['snippet'][:100]}...")

        except Exception as e:
            print(f"測試失敗: {e}")
