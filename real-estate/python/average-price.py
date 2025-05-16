import requests
import json
import os
import numpy as np

#attempted to do a linear regression here.  If i had more time I would create dummy variables for 
# city zone, and place pandas or scipy in the docker container so I can get p values, etc. 
# (had some problems setting that up)  If I had a lot more time I would 
# test better models then linear regression

json_url = 'https://raw.githubusercontent.com/bogdanfazakas/datasets/refs/heads/main/data.json'
output_folder = '/data/outputs'
output_file = os.path.join(output_folder, 'results.json')

def perform_linear_regression(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        properties = response.json()

        if not isinstance(properties, list):
            raise ValueError("Expected JSON to be a list")

        # Prepare feature and target lists
        features = []
        prices = []

        for prop in properties:
            info = prop.get("info", {})
            price = info.get("price")
            rooms_no = info.get("roomsNo")
            surface = info.get("surface")
            bathrooms_no = info.get("bathroomsNo")

            if None in (price, rooms_no, surface, bathrooms_no):
                continue

            features.append([rooms_no, surface, bathrooms_no])
            prices.append(price)

        if len(features) == 0:
            print("❌ No valid data available.")
            return

        X = np.array(features)
        y = np.array(prices)

        # Add intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Safe pseudo-inverse calculation
        theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        # Feature names for clarity
        feature_names = ["Intercept", "roomsNo", "surface", "bathroomsNo"]

        # Build results dictionary
        results = {name: float(f"{coef:.2f}") for name, coef in zip(feature_names, theta_best)}

        # Print coefficient table
        print("\nLinear Regression Coefficients:")
        for name, coef in zip(feature_names, theta_best):
            print(f"{name}: {coef:.2f}")

        # Build the pricing formula
        formula_terms = []
        for name, coef in zip(feature_names, theta_best):
            term = f"{coef:.2f} * {name}" if name != "Intercept" else f"{coef:.2f}"
            formula_terms.append(term)

        price_formula = " + ".join(formula_terms)
        price_per_room_formula = f"({price_formula}) / roomsNo"

        # Print the pricing formulas
        print("\nExpected Price Per Property:")
        print(price_formula)

        print("\nExpected Price Per Room:")
        print(price_per_room_formula)

        # Write to output file
        os.makedirs(output_folder, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n✅ Results written to:", output_file)

    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Run it
perform_linear_regression(json_url)
