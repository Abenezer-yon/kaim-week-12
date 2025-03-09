import json
from telethon import TelegramClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables for api
api_id = os.getenv('API_ID')
api_hash = os.getenv('API_HASH')

channels = [
    '@ZemenExpress',
    '@sinayelj',
    '@modernshoppingcenter',
    '@Shewabrand',
    '@helloomarketethiopia'
]

# Create a Telegram client
client = TelegramClient('telegram_data_scraper', api_id, api_hash)

async def main():
    await client.start()
    all_messages = []

    for channel in channels:
        try:
            entity = await client.get_entity(channel)
            async for message in client.iter_messages(entity):
                all_messages.append({
                    'channel': channel,
                    'sender': message.sender_id,
                    'timestamp': message.date.isoformat(),
                    'content': message.message
                })
        except Exception as e:
            print(f"Error fetching messages from {channel}: {e}")

    # Save messages to a JSON file
    with open('telegram_messages.json', 'w', encoding='utf-8') as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=4)
    print("Messages saved to telegram_messages.json")

with client:
    client.loop.run_until_complete(main())