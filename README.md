# HackIllinois Summary Bot

A Discord bot that privately summarizes recent hackathon-planning discussion
with Google's Gemini API. The `/summary` slash command reads human messages
from the current channel, filters out casual chatter, and returns an ephemeral
summary visible only to the requester.

## Features

- `/summary hours:<1-168>` slash command
- Ephemeral responses in the originating Discord channel
- Objective filtering for decisions, updates, action items, and resources
- Bot-message filtering and timezone-aware history windows
- Automatic transcript batching for busy channels
- Discord-safe chunking for summaries over 2,000 characters
- Gemini model fallback and actionable API error messages

## Requirements

- Python 3.10 or newer
- A Discord application with a bot token
- A Gemini API key from Google AI Studio

## Discord configuration

In the Discord Developer Portal:

1. Enable **Message Content Intent** on the Bot page.
2. Configure Guild Install with the `bot` and `applications.commands` scopes.
3. Grant the bot **View Channels**, **Send Messages**, and
   **Read Message History** permissions.

Private channels must grant the bot or its role access separately.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add credentials to `.env`:

```env
DISCORD_TOKEN=your_discord_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

Start the bot:

```bash
python bot.py
```

After Discord registers the command, enter `/summary` in a server channel and
provide the number of past hours to summarize. The result is ephemeral and is
not posted publicly or sent by DM.

## Security

Never commit `.env` or share either token. The included `.gitignore` excludes
local credentials, virtual environments, and Python cache files.
