from telethon.sync import TelegramClient
from telethon.tl.types import InputMessagesFilterPhotos, InputMessagesFilterDocument
import pandas as pd
import os

# Step 1: Telegram API credentials
api_id = ''
api_hash = ''
phone_number = ''  

# Step 2: Connect to the Telegram client
client = TelegramClient('ethioMartScraper', api_id, api_hash)

# List of channel usernames (you can update these)
channels = ['shageronlinestore', 'ethiomarket', 'addis_shop', 'amharic_store1', 'fashion_addis']

# Step 3: Fetch messages
async def fetch_messages(channel_username, limit=200):
    await client.start()
    messages = []
    async for msg in client.iter_messages(channel_username, limit=limit):
        messages.append({
            'channel': channel_username,
            'text': msg.message,
            'date': msg.date,
            'sender_id': msg.sender_id,
            'media': 'image' if msg.photo else 'document' if msg.document else None
        })
    return messages

# Step 4: Collect data from all channels
import asyncio

async def collect_all_data():
    all_data = []
    for ch in channels:
        data = await fetch_messages(ch)
        all_data.extend(data)
    return pd.DataFrame(all_data)

df = asyncio.run(collect_all_data())
df.dropna(subset=["text"], inplace=True)  # Remove messages without text
df.to_csv("raw_telegram_data.csv", index=False)
import re

# Load the data
df = pd.read_csv("raw_telegram_data.csv")

# Clean and tokenize Amharic text
def preprocess_amharic(text):
    # Remove emojis and symbols
    text = re.sub(r'[^\w\s፡።፣፤፥፦፧፨]+', '', str(text))
    # Normalize spaces and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(preprocess_amharic)
df_cleaned = df[['channel', 'date', 'sender_id', 'clean_text', 'media']]
df_cleaned.to_json("preprocessed_telegram_data.json", orient='records', force_ascii=False, lines=True)
