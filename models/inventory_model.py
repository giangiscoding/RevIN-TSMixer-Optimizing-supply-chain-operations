import scipy.stats as stats
import numpy as np

class ProbabilisticInventoryModel:
    def __init__(self, holding_cost, ordering_cost, shortage_cost, lead_time):
        self.h = holding_cost
        self.o = ordering_cost
        self.cs = shortage_cost
        self.L = lead_time

    def standard_normal_loss(self, z):
        pdf = stats.norm.pdf(z)
        cdf = stats.norm.cdf(z)
        return pdf - z * (1 - cdf)

    def calculate_metrics(self, forecasted_demands, demand_std):
        mu_D = np.mean(forecasted_demands)
        sigma_D = demand_std

        Q_star = np.sqrt((2 * mu_D * self.o) / self.h)
        alpha = 1 - (self.h * Q_star) / (self.cs * mu_D)
        alpha = np.clip(alpha, 1e-5, 0.99999) 
        
        z_alpha = stats.norm.ppf(alpha)
        SS = z_alpha * sigma_D * np.sqrt(self.L)
        r = mu_D * self.L + SS

        L_z = self.standard_normal_loss(z_alpha)
        E_S = L_z * sigma_D * np.sqrt(self.L)

        expected_shortage_cost = (self.cs * E_S * mu_D) / Q_star
        ordering_cost = (mu_D / Q_star) * self.o
        holding_cost = (Q_star / 2 + max(SS, 0)) * self.h
        total_cost = ordering_cost + holding_cost + expected_shortage_cost

        return {
            "Average Forecasted Demand (mu_D)": round(mu_D, 2),
            "Optimal Order Quantity (Q*)": round(Q_star, 2),
            "Service Level (alpha)": round(alpha, 4),
            "Safety Stock (SS)": round(SS, 2),
            "Reorder Point (r)": round(r, 2),
            "Total Cost (TC)": round(total_cost, 2)
        }