# Payment Risk Scorer

* A machine learning service that scores synthetic payment transactions for fraud and compliance risk
* Each score comes with a clear level and human readable reasons, so a reviewer can see why a payment looks risky

## What it does

Payment Risk Scorer builds synthetic payment data, trains a small neural network in PyTorch, and serves scores through a FastAPI service. The data is synthetic only, with a fixed seed, so every run can be repeated exactly. The model learns patterns like high amount, unusual country, and high velocity, and the API returns a score from 0 to 1, a risk level of low, medium, or high, and reason codes in plain words.

## Why it matters

Payment teams need to spot fraud and compliance risk fast, without blocking good payments. A score alone is not enough. A reviewer needs reasons to act, to check a payment, or to clear it. This project shows that flow in a small and honest form: synthetic data, a trained model with measured metrics, clear reasons, and a simple API and demo page.

## Features

* Synthetic transaction generator with a fixed seed
* PyTorch risk model with training and measured metrics
* Reason codes for high amount, unusual country, high velocity, night hour, and high risk merchant
* FastAPI service with health check and score endpoint
* Simple demo page that calls the API in a browser
* Tests with pytest that cover the generator, the model, the reasons, and the API

## Project layout

* train.py trains the model and writes metrics
* app.py runs the FastAPI service
* src/generator.py makes synthetic transactions and encodes features
* src/model.py defines the model and the metric code
* src/explain.py defines reason codes, risk levels, and scoring
* static/index.html is the demo page
* tests/test_generator.py tests the generator
* tests/test_model.py tests the model
* tests/test_explain.py tests the reasons
* tests/test_api.py tests the API
* requirements.txt lists the packages
* requirements_light.txt lists only the packages needed to run the API without PyTorch
* render.yaml is the Render blueprint for live hosting
* Dockerfile is an optional container setup for the light runtime
* src/light_model.py runs the same model with NumPy only, using the weights file
* export_weights.py converts the trained model to the weights file
* models/model.pt is the trained model, made by train.py
* models/weights.json holds the trained weights as plain JSON, made by train.py
* models/metrics.json holds the measured metrics, made by train.py

## Install

* Install Python 3.10 or newer
* Install the packages listed in requirements.txt with pip
* Use this command to install the main packages

```
pip install torch fastapi uvicorn httpx numpy pytest pydantic
```

## Train

* Run the training script from the project folder

```
python train.py
```

* Training makes synthetic data, trains the model, and writes the model and metrics
* Training also writes the light weights file, so the live API can use the same weights without PyTorch
* The metrics file holds the real results from that run, with the test set size and the fraud rate

## Run the API

* Train first, so the model file exists
* Run the API from the project folder

```
python app.py
```

* Open this address in a browser to use the demo page

```
http://localhost:8000
```

* The health check is at GET /health
* The score endpoint is POST /score
* Send amount, merchant category, country, hour, device type, and velocity
* The response has score, risk level, and reasons

## Live Hosting

* The API can run live on Render on the free tier
* The repo includes a Render blueprint file named render.yaml
* Render installs only the light requirements file, so the deploy stays small and fast
* The light path uses the same trained weights, stored as plain JSON, so scores match the PyTorch model
* The full requirements file is still used for training, which needs PyTorch
* To deploy, create a free Render account, create a new web service from the GitHub repo, and choose the blueprint file when Render asks for it
* Render will show a live web service URL on the dashboard after deploy
* Open that URL in a browser to see the demo page
* The health check lives at the live URL with health added at the end
* To point the demo page at the live API from another host, set API_BASE_URL at the top of static/index.html to the live web service URL, then redeploy the demo page
* If the demo page is served by the same API service, leave API_BASE_URL empty and it will use the same host
* The free tier may sleep after inactivity, so the first request after a pause can be slow
* The trained model file is not pushed to GitHub, because it is binary. Run train.py once after cloning to make it, or use the light weights file, which is pushed and works without PyTorch

## Run the tests

* Run the tests from the project folder

```
pytest
```

* The tests cover the generator, the model, the reasons, and the API
* All tests should pass before you share the project

## How scoring works

* The generator makes transactions with amount, merchant category, country, hour, device type, and velocity
* The encoder turns each transaction into numbers the model can use, with log amount, velocity, hour as sine and cosine, and one hot codes for merchant and device
* The model is a small neural network with two hidden layers and a single output
* The score is the model output as a number from 0 to 1
* Risk level is low below 0.30, medium from 0.30 to below 0.70, and high at 0.70 and above
* Reason codes are rules that look at the same transaction, so the reasons match what a human would check

## Reason codes

* high_amount means the amount is high for this payment
* very_high_amount means the amount is very high for this payment
* unusual_country means the country is unusual for this account pattern
* high_velocity means many transactions in a short time
* night_hour means the transaction hour is in the night period
* high_risk_merchant means the merchant category is often linked to higher risk
* no_strong_signal means no strong risk signal in this transaction

## Metrics

* Metrics are measured on the held out test set, not on the training data
* ROC AUC shows how well the score ranks fraud above good payments, with 1.0 as perfect and 0.5 as chance
* PR AUC shows how well the score finds fraud when fraud is rare, with 1.0 as perfect
* Accuracy, precision, recall, and F1 are measured at a threshold of 0.5
* See models/metrics.json for the numbers from the latest training run
* Training used 8000 transactions, 60 epochs, and seed 42
* Test set size is 1200 transactions
* Fraud rate in test set is 0.1892
* Fraud ROC AUC is 0.7350
* Fraud PR AUC is 0.4060
* Fraud accuracy is 0.6808
* Fraud precision is 0.3274
* Fraud recall is 0.6520
* Fraud F1 is 0.4359
* Compliance ROC AUC using the same score is 0.9099
* Compliance PR AUC using the same score is 0.4007
