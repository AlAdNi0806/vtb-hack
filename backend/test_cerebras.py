#!/usr/bin/env python3
"""
Test Cerebras API connection
"""

import asyncio
import httpx
from config import settings

async def test_cerebras_api():
    """Test the Cerebras API connection"""
    print("Testing Cerebras API connection...")
    print(f"API Key: {settings.CEREBRAS_API_KEY[:20]}...")
    print(f"Base URL: {settings.CEREBRAS_BASE_URL}")
    print(f"Model: {settings.CEREBRAS_MODEL}")
    
    client = httpx.AsyncClient(
        base_url=settings.CEREBRAS_BASE_URL,
        headers={
            "Authorization": f"Bearer {settings.CEREBRAS_API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=30.0
    )
    
    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": settings.CEREBRAS_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Cerebras API connection successful!")
            print(f"Response: {result}")
        else:
            print(f"❌ Cerebras API error: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
    
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(test_cerebras_api())
