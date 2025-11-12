import requests

bot_token = '7914157083:AAGCCCHsdF0RQJGF2ezNnxN0mYszRQXfNOg'
get_updates_url = f"https://api.telegram.org/bot{bot_token}/getUpdates"

res = requests.get(get_updates_url)
data = res.json()

if data['result']:
    chat_id = data['result'][-1]['message']['chat']['id']
    print(f"💬 Chat ID: {chat_id}")
    
    send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    message = "Hey there! Your bot is now active 🚀"
    payload = {'chat_id': chat_id, 'text': message}
    requests.post(send_url, data=payload)
else:
    print("😕 No messages found. Send /start to your bot on Telegram.")