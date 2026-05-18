import pandas as pd
import json
from datetime import date

def sync_prices(csv_path="products.csv", json_output="prices.json"):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Map products.csv names to prices.json keys
    mapping = {
        "Cement OPC 53-Grade": "cement_bag",
        "Cement PPC": "cement_bag_ppc",
        "Brick Awwal": "brick",
        "Ravi Sand": "sand_cft",
        "Chenab Sand": "sand_cft_fine",
        "Crush / Bajri (3/4 inch)": "gravel_cft",
        "Steel / Sarya TMT Bar 12mm": "steel_kg",
        "Emulsion Paint (Interior)": "paint_liter",
        "Ceramic Floor Tile (Local) 12x12": "tiles_sqft",
        "Deodar Wood": "wood_cft"
    }
    
    # Tier mapping
    tiers = ["Economy", "Standard", "Premium", "Luxury"]
    
    items_output = {}
    
    for item_name, price_key in mapping.items():
        # Case-insensitive substring match
        sub_df = df[df["item_name"].str.contains(item_name, case=False, na=False)]
        if sub_df.empty:
            print(f"  ? No data found for '{item_name}'")
            continue
            
        unit = sub_df.iloc[0]["unit"]
        print(f"  + Matched '{item_name}' -> {price_key}")
        
        # Get prices for each tier
        tier_prices = {}
        for tier in tiers:
            tier_df = sub_df[sub_df["quality_grade"].str.contains(tier, case=False, na=False)]
            if not tier_df.empty:
                tier_prices[tier.lower()] = int(tier_df["final_price_pkr"].mean())
            else:
                tier_prices[tier.lower()] = int(sub_df["final_price_pkr"].mean())
        
        items_output[price_key] = {
            "unit": unit,
            "economy": tier_prices["economy"],
            "standard": tier_prices["standard"],
            "premium": tier_prices["premium"],
            "luxury": tier_prices["luxury"]
        }
    
    # ... labour etc
    items_output["labour_sqft"] = {"unit": "sqft", "economy": 350, "standard": 550, "premium": 850, "luxury": 1200}
    items_output["plumbing_lumpsum"] = {"unit": "washroom", "economy": 45000, "standard": 75000, "premium": 120000, "luxury": 250000}
    items_output["electrical_lumpsum"] = {"unit": "kitchen", "economy": 35000, "standard": 60000, "premium": 110000, "luxury": 220000}

    output_data = {
        "last_updated": str(date.today()),
        "items": items_output
    }
    
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"DONE: Saved {len(items_output)} items to {json_output}")

if __name__ == "__main__":
    sync_prices()
