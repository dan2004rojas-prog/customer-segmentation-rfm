SELECT
    customer_id,
    CURRENT_DATE - MAX(order_date)::DATE                        AS recency_days,
    COUNT(DISTINCT order_id)                                    AS frequency,
    SUM(order_amount)                                           AS monetary
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY customer_id
HAVING SUM(order_amount) > 0
ORDER BY monetary DESC;
