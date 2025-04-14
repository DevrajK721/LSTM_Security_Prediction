#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <tuple>
#include <stdexcept>
#include <string>
#include <algorithm>  // for std::min, std::max

namespace py = pybind11;

//------------------------------------------------------------------------------
// Existing Autocorrelation functions (unchanged)
//------------------------------------------------------------------------------
std::vector<double> compute_autocorrelations(py::array_t<double> returns, int max_lag) {
    auto r = returns.unchecked<1>();
    int n = r.shape(0);
    std::vector<double> autocorrs(max_lag + 1);
    
    double mean = 0.0;
    for (int i = 0; i < n; i++)
        mean += r(i);
    mean /= n;
    
    double variance = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = r(i) - mean;
        variance += diff * diff;
    }
    
    for (int lag = 0; lag <= max_lag; lag++) {
        double numerator = 0.0;
        for (int i = lag; i < n; i++)
            numerator += (r(i) - mean) * (r(i - lag) - mean);
        autocorrs[lag] = numerator / variance;
    }
    
    return autocorrs;
}

py::array_t<double> calculate_return_autocorrelations(py::object df, int max_lag) {
    py::array_t<double> returns = df.attr("Target").cast<py::array_t<double>>();
    std::vector<double> autocorrs = compute_autocorrelations(returns, max_lag);
    auto result = py::array_t<double>(autocorrs.size());
    auto buf = result.request();
    double *ptr = static_cast<double *>(buf.ptr);
    for (size_t i = 0; i < autocorrs.size(); i++)
        ptr[i] = autocorrs[i];
    return result;
}

//------------------------------------------------------------------------------
// ARIMA-GARCH Grid Search Implementation
//------------------------------------------------------------------------------
//
// Helper function to fit an AR model of order p to vector x.
// Returns (params, residuals, certainty) where for p>=1:
//   - params[0] is intercept and params[1..p] are AR coefficients.
//   - For an AR(1) model, "certainty" is defined as 1 - p-value computed from the t-statistic
//     of the AR coefficient. For higher order models, we use R² as a fallback.
std::tuple<std::vector<double>, std::vector<double>, double> fit_ar_model(const std::vector<double>& x, int p) {
    int n = x.size();
    if(n <= p)
        throw std::runtime_error("Not enough data points for AR(" + std::to_string(p) + ") model");

    std::vector<double> params;
    std::vector<double> residuals;
    double certainty = 0.0;  // will be computed below
    
    // Build the regression design matrix:
    int N = n - p;
    std::vector<std::vector<double>> X(N, std::vector<double>(p+1, 1.0)); // first column constant (intercept)
    std::vector<double> Y(N);
    for (int t = p; t < n; t++) {
        Y[t-p] = x[t];
        for (int j = 1; j <= p; j++) {
            X[t-p][j] = x[t-j];
        }
    }
    std::vector<double> beta(p+1, 0.0);
    
    if(p == 1) {
        // AR(1) closed-form solution
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        for (int i = 0; i < N; i++) {
            double xlag = X[i][1];
            double yval = Y[i];
            sum_x += xlag;
            sum_y += yval;
            sum_xy += xlag * yval;
            sum_x2 += xlag * xlag;
        }
        double denom = N * sum_x2 - sum_x * sum_x;
        beta[1] = (N * sum_xy - sum_x * sum_y) / (denom + 1e-8);
        beta[0] = (sum_y - beta[1] * sum_x) / N;
    } else if (p == 2) {
        // Solve 3x3 system using Cramer's rule
        double S00 = 0, S01 = 0, S02 = 0;
        double S11 = 0, S12 = 0, S22 = 0;
        double T0 = 0, T1 = 0, T2 = 0;
        for (int i = 0; i < N; i++) {
            double x0 = X[i][0]; // 1.0
            double x1 = X[i][1];
            double x2 = X[i][2];
            double y = Y[i];
            S00 += x0 * x0; S01 += x0 * x1; S02 += x0 * x2;
            S11 += x1 * x1; S12 += x1 * x2; S22 += x2 * x2;
            T0 += x0 * y; T1 += x1 * y; T2 += x2 * y;
        }
        double det = S00*(S11*S22 - S12*S12) - S01*(S01*S22 - S12*S02) + S02*(S01*S12 - S11*S02);
        if(std::abs(det) < 1e-8) det = 1e-8;
        beta[0] = (T0*(S11*S22 - S12*S12) - S01*(T1*S22 - S12*T2) + S02*(T1*S12 - S11*T2)) / det;
        beta[1] = (S00*(T1*S22 - S12*T2) - T0*(S01*S22 - S12*S02) + S02*(S01*T2 - T1*S02)) / det;
        beta[2] = (S00*(S11*T2 - T1*S12) - S01*(S01*T2 - T0*S12) + T0*(S01*S12 - S11*S02)) / det;
    }
    
    // Compute fitted values and residuals.
    residuals.resize(N);
    double ssRes = 0.0, ssTot = 0.0;
    double meanY = 0.0;
    for (double y : Y) meanY += y;
    meanY /= N;
    for (int i = 0; i < N; i++) {
        double y_pred = beta[0];
        for (int j = 1; j <= p; j++) {
            y_pred += beta[j] * X[i][j];
        }
        residuals[i] = Y[i] - y_pred;
        ssRes += residuals[i] * residuals[i];
        double diff = Y[i] - meanY;
        ssTot += diff * diff;
    }
    
    // Compute certainty measure.
    if(p == 1) {
        // Compute variance estimate s^2 for residuals:
        double s2 = ssRes / (N - 2);
        // Compute Sxx for the lagged predictor.
        double sum_xlag = 0.0;
        for (int i = 0; i < N; i++) {
            sum_xlag += X[i][1];
        }
        double mean_xlag = sum_xlag / N;
        double Sxx = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = X[i][1] - mean_xlag;
            Sxx += diff * diff;
        }
        double se = std::sqrt(s2 / (Sxx + 1e-8));
        double t_stat = beta[1] / se;
        // Using the complementary error function to approximate a two-sided p-value.
        double p_val = std::erfc(std::fabs(t_stat)/std::sqrt(2.0));
        // Define certainty as 1 - p_value (clip to [0,1]).
        certainty = std::max(0.0, std::min(1.0, 1.0 - p_val));
    } else {
        // For p==2, use the R² measure as a fallback.
        double r2 = (ssTot == 0) ? 1.0 : 1.0 - ssRes/ssTot;
        certainty = r2;
    }
    params = beta;
    return std::make_tuple(params, residuals, certainty);
}

// A simple function to fit GARCH(1,1) to residuals using a grid search.
std::tuple<double, double, double> fit_garch11(const std::vector<double>& r) {
    double best_ll = -std::numeric_limits<double>::infinity();
    double best_omega = 1e-6, best_alpha = 0.1, best_beta = 0.85;
    for (double omega : {1e-6, 5e-6}) {
        for (double alpha : {0.05, 0.1, 0.15}) {
            for (double beta : {0.8, 0.85, 0.9}) {
                double sigma2 = omega / (1 - alpha - beta);
                double ll = 0.0;
                for (double e : r) {
                    ll += -0.5 * (std::log(2 * M_PI * sigma2) + (e * e) / sigma2);
                    sigma2 = omega + alpha * e * e + beta * sigma2;
                }
                if (ll > best_ll) {
                    best_ll = ll;
                    best_omega = omega;
                    best_alpha = alpha;
                    best_beta = beta;
                }
            }
        }
    }
    return std::make_tuple(best_omega, best_alpha, best_beta);
}

// Compute log-likelihood for ARIMA-GARCH given residuals and GARCH parameters.
double compute_loglikelihood(const std::vector<double>& r, double omega, double alpha, double beta) {
    double ll = 0.0;
    double sigma2 = omega / (1 - alpha - beta);
    for (double e : r) {
        ll += -0.5 * (std::log(2 * M_PI * sigma2) + (e * e)/sigma2);
        sigma2 = omega + alpha * e * e + beta * sigma2;
    }
    return ll;
}

//------------------------------------------------------------------------------
// Grid Search for Optimal ARIMA-GARCH Model (p in {1,2} only)
//------------------------------------------------------------------------------
py::dict grid_search_arima_garch(py::array_t<double> log_returns_np) {
    auto buf = log_returns_np.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    std::vector<double> log_returns(ptr, ptr + n);
    
    double best_aic = std::numeric_limits<double>::infinity();
    int best_p = -1;
    std::vector<double> best_arima_params;
    std::vector<double> best_residuals;
    double best_certainty = 0.0;
    double best_omega, best_alpha, best_beta;
    double best_ll = 0.0;
    
    // Force AR orders 1 and 2.
    for (int p = 1; p <= 2; p++) {
        std::vector<double> params;
        std::vector<double> residuals;
        double certainty = 0.0;
        try {
            std::tie(params, residuals, certainty) = fit_ar_model(log_returns, p);
        } catch(const std::exception& ex) {
            continue;
        }
        double omega, alpha, beta;
        std::tie(omega, alpha, beta) = fit_garch11(residuals);
        
        double ll = compute_loglikelihood(residuals, omega, alpha, beta);
        int num_params = p + 1 + 3;
        double aic = -2 * ll + 2 * num_params;
        
        if(aic < best_aic) {
            best_aic = aic;
            best_p = p;
            best_arima_params = params;
            best_residuals = residuals;
            best_certainty = certainty;
            best_omega = omega;
            best_alpha = alpha;
            best_beta = beta;
            best_ll = ll;
        }
    }
    if(best_p == -1)
        throw std::runtime_error("ARIMA-GARCH model fitting failed for all candidate AR orders.");
    
    py::dict results;
    results["arima_order"] = py::make_tuple(best_p, 0, 0);  // d=0, q=0 in our simplified model.
    results["arima_params"] = best_arima_params;
    results["garch_params"] = py::make_tuple(best_omega, best_alpha, best_beta);
    results["aic"] = best_aic;
    results["certainty"] = best_certainty;
    results["loglikelihood"] = best_ll;
    
    return results;
}

//------------------------------------------------------------------------------
// Prediction Function Using the Fitted ARIMA-GARCH Model
//------------------------------------------------------------------------------
double predict_next_return(py::dict model, py::array_t<double> recent_returns_np) {
    std::vector<double> arima_params = model["arima_params"].cast<std::vector<double>>();
    auto buf = recent_returns_np.request();
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    if(n == 0)
        throw std::runtime_error("Recent returns array is empty");
    double last_return = ptr[n-1];
    double forecast = 0.0;
    if(arima_params.size() == 1)
        forecast = arima_params[0];
    else
        forecast = arima_params[0] + arima_params[1] * last_return;
    return forecast;
}

//------------------------------------------------------------------------------
// PyBind Module Definition
//------------------------------------------------------------------------------
PYBIND11_MODULE(QuantCoreStats, m) {
    m.doc() = "QuantCore Statistics Functions including grid search for ARIMA-GARCH models and prediction";

    m.def("compute_autocorrelations", &compute_autocorrelations, "Compute the autocorrelations of a given time series of log-daily returns up to a specified maximum lag.");
    
    m.def("calculate_return_autocorrelations", &calculate_return_autocorrelations, "Calculate autocorrelations from a DataFrame's 'Target' column.");
    
    m.def("grid_search_arima_garch", &grid_search_arima_garch, "Perform a grid search over AR orders (1,2) and fit a GARCH(1,1) on the residuals; returns a dictionary with optimal model parameters, AIC, and a certainty measure.");
    
    m.def("predict_next_return", &predict_next_return, "Make a one-step-ahead prediction using a fitted ARIMA-GARCH model. Input the model dictionary and a numpy array of recent log returns.");
}