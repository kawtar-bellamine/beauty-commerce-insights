-- Staging: typed, renamed, de-duplicated copies of the raw marketplace tables.
-- Raw files are read as text first so that no value is silently converted.

CREATE OR REPLACE TABLE stg_orders AS
SELECT
    order_id,
    customer_id,
    order_status                                   AS status,
    TRY_CAST(order_purchase_timestamp AS TIMESTAMP)      AS purchased_at,
    TRY_CAST(order_approved_at AS TIMESTAMP)             AS approved_at,
    TRY_CAST(order_delivered_carrier_date AS TIMESTAMP)  AS shipped_at,
    TRY_CAST(order_delivered_customer_date AS TIMESTAMP) AS delivered_at,
    TRY_CAST(order_estimated_delivery_date AS TIMESTAMP) AS estimated_at
FROM raw_orders;

CREATE OR REPLACE TABLE stg_items AS
SELECT
    order_id,
    CAST(order_item_id AS INTEGER) AS item_seq,
    product_id,
    seller_id,
    CAST(price AS DOUBLE)          AS price,
    CAST(freight_value AS DOUBLE)  AS freight
FROM raw_items;

CREATE OR REPLACE TABLE stg_products AS
SELECT
    p.product_id,
    COALESCE(p.product_category_name, 'unknown')                    AS category_pt,
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
    TRY_CAST(p.product_weight_g AS DOUBLE)                          AS weight_g,
    TRY_CAST(p.product_photos_qty AS INTEGER)                       AS photos,
    TRY_CAST(p.product_description_lenght AS INTEGER)               AS description_length
FROM raw_products p
LEFT JOIN raw_category_translation t USING (product_category_name);

CREATE OR REPLACE TABLE stg_customers AS
SELECT customer_id, customer_unique_id, customer_zip_code_prefix AS zip_prefix,
       customer_city AS city, customer_state AS state
FROM raw_customers;

CREATE OR REPLACE TABLE stg_sellers AS
SELECT seller_id, seller_state FROM raw_sellers;

-- An order can carry several reviews: keep the most recent answer.
CREATE OR REPLACE TABLE stg_reviews AS
SELECT order_id, review_score, comment_title, comment_message, answered_at
FROM (
    SELECT order_id,
           CAST(review_score AS INTEGER)                     AS review_score,
           review_comment_title                              AS comment_title,
           review_comment_message                            AS comment_message,
           TRY_CAST(review_answer_timestamp AS TIMESTAMP)    AS answered_at,
           ROW_NUMBER() OVER (PARTITION BY order_id
                              ORDER BY TRY_CAST(review_answer_timestamp AS TIMESTAMP) DESC) AS rn
    FROM raw_reviews
) WHERE rn = 1;

CREATE OR REPLACE TABLE stg_payments AS
SELECT order_id,
       SUM(CAST(payment_value AS DOUBLE))                         AS paid,
       MAX(CAST(payment_installments AS INTEGER))                 AS installments,
       ARG_MAX(payment_type, CAST(payment_value AS DOUBLE))       AS main_payment
FROM raw_payments
GROUP BY order_id;
