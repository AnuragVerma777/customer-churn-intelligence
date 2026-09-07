-- Executed against the SQLite customers table (identifiers are selected after schema detection).
SELECT COUNT(*) AS total_customers FROM customers;
SELECT SUM(CASE WHEN churn_target = 1 THEN 1 ELSE 0 END) AS churned_customers,
       AVG(churn_target) AS churn_rate FROM customers;
SELECT segment, COUNT(*) AS customers, AVG(revenue) AS average_revenue,
       AVG(churn_target) AS churn_rate FROM customers GROUP BY segment;
SELECT customer_id, revenue, churn_probability,
       revenue * churn_probability AS revenue_at_risk
FROM customer_scores ORDER BY revenue_at_risk DESC;
