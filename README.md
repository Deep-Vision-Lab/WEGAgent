# 1. Install dependencies
pip install -r requirements.txt

# 2. verify OpenAI API key is available
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"

# 3. Crawl iFixit guides (creates pre-WEG data)
python -m weg_agent.crawl --device refrigerators --limit 1

# 4. Run deterministic WEG builder (no agent)
python -m weg_agent.run --root preweg_data/appliances --no-resume

# 5. Run AI multitool agent LangChain based
python -m weg_agent.run_langchain --root preweg_data/appliances
