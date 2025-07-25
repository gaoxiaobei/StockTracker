#!/usr/bin/env python3
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import data.fetcher as data_fetcher
import models.advanced as advanced_model

def test_transformer_model():
    """Test the Transformer model specifically"""
    print("Testing Transformer model...")
    
    try:
        # Get stock data
        print("Fetching stock data...")
        stock_data = data_fetcher.get_stock_data('000001', period='daily', start_date='20240101', adjust='qfq')
        
        if stock_data.empty:
            print("Failed to get stock data")
            return False
            
        print(f"Got {len(stock_data)} records")
        
        # Create predictor
        print("Creating AdvancedStockPredictor with transformer model...")
        predictor = advanced_model.AdvancedStockPredictor(look_back=60, model_type='transformer')
        
        # Train model
        print("Training transformer model...")
        history = predictor.train(stock_data, epochs=5, batch_size=32)  # Use fewer epochs for testing
        print("Training completed successfully")
        
        # Predict
        print("Making prediction...")
        predicted_price = predictor.predict(stock_data)
        print(f"Predicted price: {predicted_price}")
        
        return True
        
    except Exception as e:
        print(f"Error testing transformer model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transformer_model()
    if success:
        print("✅ Transformer model test passed")
    else:
        print("❌ Transformer model test failed")