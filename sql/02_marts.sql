-- Marts: a small star schema for analysis and the dashboard.
-- "Beauty" = health_beauty + perfumery, the beauty categories of the marketplace.

CREATE OR REPLACE TABLE dim_product AS
SELECT *, category IN ('health_beauty', 'perfumery') AS is_beauty FROM stg_products;

CREATE OR REPLACE TABLE fct_order_item AS
SELECT i.order_id, i.item_seq, i.product_id, i.seller_id, i.price, i.freight,
       p.category, p.is_beauty, s.seller_state
FROM stg_items i
JOIN dim_product p USING (product_id)
LEFT JOIN stg_sellers s USING (seller_id);

-- One row per order, with what the customer experienced.
CREATE OR REPLACE TABLE fct_orders AS
WITH items AS (
    SELECT order_id,
           COUNT(*)                                  AS n_items,
           COUNT(DISTINCT seller_id)                 AS n_sellers,
           SUM(price)                                AS gmv,
           SUM(freight)                              AS freight,
           SUM(price) FILTER (WHERE is_beauty)       AS beauty_gmv,
           SUM(price) FILTER (WHERE category = 'perfumery') AS perfume_gmv,
           BOOL_OR(is_beauty)                        AS has_beauty,
           ARG_MIN(seller_state, item_seq)           AS seller_state
    FROM fct_order_item GROUP BY order_id
)
SELECT o.order_id, c.customer_unique_id, c.state AS customer_state, o.status,
       o.purchased_at, o.shipped_at, o.delivered_at, o.estimated_at,
       i.n_items, i.n_sellers, i.gmv, i.freight, COALESCE(i.beauty_gmv, 0) AS beauty_gmv,
       COALESCE(i.perfume_gmv, 0) AS perfume_gmv, COALESCE(i.has_beauty, FALSE) AS has_beauty,
       i.seller_state, i.seller_state = c.state AS same_state,
       DATE_DIFF('hour', o.estimated_at, o.delivered_at) / 24.0  AS delay_days,
       DATE_DIFF('hour', o.purchased_at, o.delivered_at) / 24.0  AS delivery_days,
       DATE_DIFF('hour', o.purchased_at, o.estimated_at) / 24.0  AS promised_days,
       o.delivered_at > o.estimated_at                           AS is_late,
       r.review_score, r.comment_message,
       pay.paid, pay.installments, pay.main_payment
FROM stg_orders o
JOIN stg_customers c USING (customer_id)
LEFT JOIN items i USING (order_id)
LEFT JOIN stg_reviews r USING (order_id)
LEFT JOIN stg_payments pay USING (order_id);

-- Daily beauty sales on orders that were not cancelled or unavailable.
CREATE OR REPLACE TABLE daily_beauty_sales AS
SELECT CAST(purchased_at AS DATE) AS day, SUM(beauty_gmv) AS gmv, COUNT(*) AS orders
FROM fct_orders
WHERE has_beauty AND status NOT IN ('canceled', 'unavailable')
GROUP BY 1 ORDER BY 1;
