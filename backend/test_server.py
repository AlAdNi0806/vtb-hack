#!/usr/bin/env python3
"""
Simple test script to verify the backend server is working correctly.
This script tests the WebSocket connection and basic functionality.
"""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_connection():
    """Test WebSocket connection to the backend server"""
    uri = "ws://localhost:8000/ws/test_client"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("✓ WebSocket connection established")
            
            # Test start recording message
            start_message = {"type": "start_recording"}
            await websocket.send(json.dumps(start_message))
            logger.info("✓ Sent start recording message")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                logger.info(f"✓ Received response: {data}")
            except asyncio.TimeoutError:
                logger.warning("⚠ No response received within 5 seconds")
            
            # Test stop recording message
            stop_message = {"type": "stop_recording"}
            await websocket.send(json.dumps(stop_message))
            logger.info("✓ Sent stop recording message")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                logger.info(f"✓ Received response: {data}")
            except asyncio.TimeoutError:
                logger.warning("⚠ No response received within 5 seconds")
                
    except ConnectionRefusedError:
        logger.error("✗ Connection refused - make sure the backend server is running on port 8000")
        return False
    except Exception as e:
        logger.error(f"✗ WebSocket test failed: {e}")
        return False
    
    return True

async def test_health_endpoint():
    """Test the health endpoint"""
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health') as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ Health endpoint working: {data}")
                    return True
                else:
                    logger.error(f"✗ Health endpoint returned status {response.status}")
                    return False
    except Exception as e:
        logger.error(f"✗ Health endpoint test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("Starting backend server tests...")
    
    # Test health endpoint
    logger.info("\n1. Testing health endpoint...")
    health_ok = await test_health_endpoint()
    
    # Test WebSocket connection
    logger.info("\n2. Testing WebSocket connection...")
    websocket_ok = await test_websocket_connection()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"Health endpoint: {'✓ PASS' if health_ok else '✗ FAIL'}")
    logger.info(f"WebSocket connection: {'✓ PASS' if websocket_ok else '✗ FAIL'}")
    
    if health_ok and websocket_ok:
        logger.info("\n🎉 All tests passed! Backend server is working correctly.")
        return True
    else:
        logger.error("\n❌ Some tests failed. Check the backend server.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        exit(1)
