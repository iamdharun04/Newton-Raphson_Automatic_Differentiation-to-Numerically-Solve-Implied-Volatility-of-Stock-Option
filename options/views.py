import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from io import BytesIO
import base64
# Function to compute the option price using the binomial model
def binomial_option_price(S, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize option values at maturity
    option_values = np.zeros((N + 1, N + 1))

    for j in range(N + 1):
        option_values[N, j] = max(0, S * (u ** j) * (d ** (N - j)) - K)

    # Backward induction to compute option values at earlier nodes
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[i, j] = np.exp(-r * dt) * (
                p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1]
            )

    return option_values, u, d

# Objective function to find implied volatility
def implied_volatility(sigma, option_price, S, K, T, r, N):
    option_price_model, _, _ = binomial_option_price(S, K, T, r, sigma, N)
    return option_price_model[0, 0] - option_price

# Compute implied volatility using Newton-Raphson method
def compute_implied_volatility(option_price, S, K, T, r, initial_guess, N):
    obj_func = lambda sigma: implied_volatility(sigma, option_price, S, K, T, r, N)
    implied_vol = fsolve(obj_func, initial_guess)
    return implied_vol[0]

# Calculate Mean Squared Error
def calculate_mse(option_price, option_price_model):
    return np.mean((option_price_model - option_price) ** 2)

# View for homepage
def home(request):
    context = {
        'project_title': 'Using the Newton-Raphson Method with Automatic Differentiation to Numerically Solve Implied Volatility of Stock Option through Binomial Model',
        'project_intro': '''
            This project aims to address the complex problem of calculating the implied volatility of stock options.
            Implied volatility is a crucial metric in financial markets as it represents the market's view of the
            likelihood of changes in a given security's price. Traditional methods can be computationally intensive
            and may not converge efficiently. To overcome these challenges, we implement the Newton-Raphson method
            augmented with automatic differentiation.

            The core of this project utilizes the binomial model for option pricing. The binomial model is a versatile
            and widely used method that breaks down the option's life into multiple time intervals or steps. At each
            step, the price can move up or down with certain probabilities. This step-by-step approach allows for
            detailed modeling of price movements and provides a robust framework for numerical solutions.

            By integrating automatic differentiation, we significantly improve the efficiency and accuracy of the
            Newton-Raphson method. This combination allows us to swiftly and precisely solve for implied volatility,
            making it a valuable tool for traders, risk managers, and financial analysts. Our approach ensures that
            the calculations are not only accurate but also computationally feasible, even for large datasets or
            real-time applications.

            This project demonstrates the application of advanced mathematical techniques to solve real-world financial
            problems, highlighting the intersection of finance, mathematics, and computer science.
        ''',
    }
    return render(request, 'options/home.html', context)

# View for option calculation
def calculate_option(request):
    csv_path = "C:/Users/Admin/Downloads/bse30_options_data.csv"  # Update this path to your actual file location
    df = pd.read_csv(csv_path)

    # Extract the list of companies from the 'Ticker' column
    companies = df['Ticker'].tolist()

    if request.method == 'POST':
        # Get user input
        selected_company = request.POST.get('company_name')

        # Extract the necessary data for the selected company
        selected_data = df[df['Ticker'] == selected_company].iloc[0]
        S = selected_data['Close']
        K = selected_data['Strike Price']
        T = selected_data['Time to Maturity']
        option_price = selected_data['Option Price']
        r = 0.05  # Risk-free interest rate (assumed to be 5%)
        initial_guess = 0.2
        N = 100

        # Perform calculations and prepare results
        option_price_model, u, d = binomial_option_price(S, K, T, r, initial_guess, N)
        implied_vol = compute_implied_volatility(option_price, S, K, T, r, initial_guess, N)
        mse = calculate_mse(option_price, option_price_model[0, 0])

        # Generate and save plot
        fig, ax = plt.subplots()
        ax.plot(range(N + 1), option_price_model[:, 0], label='Option Price')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Option Price')
        ax.set_title(f'Option Price Over Time for {selected_company}')
        ax.legend()

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Prepare context for template rendering
        context = {
            'companies': companies,
            'selected_company': selected_company,
            'implied_vol': implied_vol,
            'mse': mse,
            'option_price_model': option_price_model,
            'u': u,
            'd': d,
            'plot_image': image_base64,
        }

        return render(request, 'options/calculate_option.html', context)

    else:
        # Handle GET request (initial form display)
        context = {
            'companies': companies,
        }
        return render(request, 'options/calculate_option.html', context)

def project_details(request):
    context = {
        'project_name': 'Option Pricing Project',
        'description': 'This project is designed to calculate option prices using the binomial model and to compute implied volatility. The application uses a dataset of options for various companies, allowing users to select a company and perform calculations on its option data.',
        'technologies_used': [
            'Python 3.11.4',
            'Django 5.0.6',
            'Pandas',
            'NumPy',
            'SciPy',
            'Matplotlib'
        ],
        'features': [
            'Calculate option prices using the binomial model',
            'Compute implied volatility',
            'Display mean squared error of the model',
            'Plot option prices over time',
            'Interactive selection of companies from a dropdown menu'
        ],
        'author': 'Your Name',
        'contact_email': 'your.email@example.com'
    }
    return render(request, 'options/project_details.html', context)
