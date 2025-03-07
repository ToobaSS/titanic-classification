import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots

# Page Title & Sidebar
st.title("Analysis of Customer Reviews on an E-Commerce Site")
st.sidebar.title("Contents")
pages = ["Introduction", "Data Exploration", "Data Visualization", "Conclusions"]
page = st.sidebar.radio("Navigate to", pages)

# Load Data
df_head10 = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\df.csv')
year_data = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\yearly_vol.csv')
weekly_data = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\weekday_vol.csv')
hourly_data = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\hour_vol.csv')
revenue_data = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\cum_sales_2017.csv')    
score_distribution = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\score_distribution.csv')
#avg_delivery_time = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\review_duration.csv')
avg_price = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\review_price.csv')
health_vs_hygiene = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\health_vs_hygiene.csv')
photos = pd.read_csv('C:\Tooba\DataScientest\Streamlit\streamlit_steps_of_a_project(1)\data\product_photos_qty.csv')


if page == "Introduction":
    st.image('https://assets-datascientest.s3.eu-west-1.amazonaws.com/mc5_da_streamlit/ecom_image.png', width=350)
    st.markdown("""
        ### Introduction  
        Welcome to our **analysis of customer reviews and ratings**. This interactive project explores the key steps involved in analyzing review data, from data extraction to visualization, to uncover meaningful insights.  
        Our objective is to understand the factors that influence customer ratings on an e-commerce platform. Customer reviews play a crucial role in shaping trust and driving purchases. By examining patterns in review scores, shipping times, and pricing, we aim to identify trends that can help improve product ratings, enhance customer satisfaction, and ultimately boost sales.  
        
        Use this dashboard to explore the data dynamically and gain actionable insights! 🚀
    """)
   
if page == "Data Exploration":
    st.header("Data Exploration")
    
    st.markdown("""
        This section presents a clear overview of the dataset’s variables, including their definitions, data types, 
        and significance. Understanding these variables helps interpret trends, correlations, and patterns in the data. 
        A structured table or list ensures clarity and quick reference.
    """)
    if st.checkbox("Display Variables & Their Meaning"):
        variables = {
            "Variables Name": ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date", "payment_value", "review_id", "review_score", "review_creation_date", "order_item_id", "product_id", "seller_id", "price", "freight_value", "product_category_name", "product_photos_qty", "customer_zip_code_prefix", "customer_city", "customer_state"],
            "Meaning": ["Order ID", "Customer ID", "Order status", "Purchase date", "Actual delivery date", "Estimated delivery date", "Transaction value", "Review ID", "Customer review score", "Review creation date", "Item number in order", "Item ID", "Seller ID", "Item price", "Shipping cost", "Product category", "Number of product images", "Customer zip code", "Customer city", "Customer state"]
        }
        st.dataframe(pd.DataFrame(variables))
    
    if st.checkbox("Display Dataset"):
        st.dataframe(df_head10)

    st.header("SQL Queries")
    st.markdown("""
    The image illustrates the database schema used to build the OLAP database for this project. 
    It defines tables, relationships, and data structures essential for analytical processing.
    """)

    st.image('https://assets-datascientest.s3.eu-west-1.amazonaws.com/architecture_bdd.png')
    
    
    st.markdown("""
        ### Feature Transformation for Visualization  

        We now introduce derived variables created to enhance the analysis of customer reviews. These transformations provide deeper insights into purchasing behavior and delivery performance. Examples include:  

        - **`delivery_duration`**: Time elapsed between purchase and delivery.  
        - **`number_review_before`**: Number of reviews submitted by the customer prior to the purchase.  
        - **`review_score_before`**: The item's average review score before the purchase.  
        - **`delivery_delay`**: The difference in days between the actual delivery date (`order_delivered_customer_date`) and the estimated delivery date.    
        
        Next, we provide the reference code used to compute these variables:  
    """)

    st.code("""
    df['delivery_duration'] = df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    df['number_review_before'] = df.groupby('product_id').cumcount()
    df['review_score_before'] = df.groupby('product_id')['review_score'].cumsum().shift() / df['number_review_before']
    df['delivery_delay'] = df['order_delivered_customer_date'] - df['order_estimated_delivery_date']
    """)
    
    st.markdown("During this analysis, we encountered a column in Portuguese, which we translated to ensure a better understanding. Below is an example of these translations.")
    st.dataframe(pd.read_csv("data/product_category_name_translation.csv"))

elif page == "Data Visualization":

    st.markdown("""
        ### Data Visualization
        In this section, we present three key data visualizations that provide insights into the project's core metrics. 
        These visualizations focus on order volume trends, revenue analysis, and review score distributions. 
        Together, they offer a comprehensive view of customer behavior and business performance.
    """)
    col1, col2, col3 = st.columns(3)

    if col1.button("Order Volume"):        
        fig1 = px.line(year_data, x='month', y='num_orders', color='year', title='Order Volume in 2017 and 2018')
        fig1 = fig1.update_layout(xaxis_title="Month", yaxis_title="Monthly Orders")
        st.plotly_chart(fig1)
        
        st.markdown("""
            The data shows a growing trend in 2017, stabilizing in 2018, with a gap in 2016 and late-2018 data. 
            This trend highlights opportunities for further growth in 2017 and areas where the ecommerce site could improve data collection in subsequent years. 
            Understanding these fluctuations can help optimize marketing strategies and inventory management for better customer engagement and sales performance.
            """)

        # Define the correct weekday order
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        fig3a = px.bar(weekly_data, x='weekday', y='num_orders', title='Weekly Sales')
        fig3a = fig3a.update_layout(
            xaxis_title="Weekday",
            yaxis_title="Average Orders",
            xaxis=dict(categoryorder='array', categoryarray=weekday_order)
        )
        st.plotly_chart(fig3a)
        st.markdown("""
            The data reveals that weekdays generate higher sales compared to weekends, with weekend sales being approximately 25% lower. 
            This insight suggests that targeted promotions or discounts during weekends could help boost sales and optimize the ecommerce site's revenue potential. 
            Understanding this pattern can also inform staffing and operational decisions to better align with customer purchasing behavior.
            Additionally, leveraging this pattern could open opportunities for targeted advertising or upselling strategies on weekends, driving additional revenue and improving overall site performance.
        """)
        
        fig3b = px.bar(hourly_data, x='hour', y='num_orders', title='Hourly Sales')
        fig3b = fig3b.update_layout(
            xaxis_title="Hour",
            yaxis_title="Average Orders",
            xaxis=dict(categoryorder='array', categoryarray=weekday_order)
        )
        st.plotly_chart(fig3b)
        st.markdown("""        
            The data shows peak sales hours occurring between 10:00 AM and 10:00 PM. 
            This insight can help optimize marketing campaigns, targeted promotions, and customer engagement strategies during these high-traffic periods. 
            Additionally, understanding these peak hours allows for more efficient resource allocation, such as staffing or site performance adjustments, to maximize revenue during prime sales times.
       """)
    
    if col2.button("Revenue Analysis"):
        revenue_data['cumulative_sales'] = revenue_data['payment_value'].cumsum()
        
        fig4 = px.bar(revenue_data, x='month', y='cumulative_sales', title="Cumulative Sales in 2017")
        fig4 = fig4.update_layout(xaxis_title="Month", yaxis_title="Cumulative Sales")
        st.plotly_chart(fig4)
                
        df_best_days = pd.read_csv('data/best_seller_sales.csv')
        fig5 = px.bar(df_best_days, x = [1,2,3,4,5], y = 'payment_value', title='Top Seller Revenue Trends')
        fig5 = fig5.update_layout(xaxis_title="Top 5 Sellers", yaxis_title="Payment Value")
        st.plotly_chart(fig5)
        
        st.markdown("""        
            E-commerce sales reached an impressive  \$ 9M in 2017 and top sellers have the potential to generate up to $100,000 per day, showcasing the immense earning power of high-performing products. 
            This insight underscores the importance of identifying and promoting top sellers to maximize revenue. 
            By leveraging data on these products, the e-commerce site can further optimize inventory, marketing strategies, and customer engagement to drive even greater sales.
        """)

    
    if col3.button("Review Score Analysis"):
        fig6 = px.bar(score_distribution, x='review_score', y='count', title="Review Score Distribution")
        fig6 = fig6.update_layout(xaxis_title="Review Score", yaxis_title="Count")
        st.plotly_chart(fig6)        
        st.markdown("""        
        A class imbalance is observed in the review scores, with 5-star ratings being significantly overrepresented. 
        This imbalance may skew the overall sentiment analysis and could impact decision-making processes. 
        Addressing this issue by considering weighting or re-sampling techniques can provide a more accurate reflection of customer feedback and improve the analysis of product performance.
        """)
                
        fig7a = px.bar(x=avg_delivery_time['review_score'], y=avg_delivery_time['delivery_duration'], title="Avg. Shipping Time")
        fig7a = fig7a.update_layout(xaxis_title="Review Score", yaxis_title="Delivery Duration")
        st.plotly_chart(fig7a)
        st.markdown("""        
        The data shows that average shipping times tend to increase with lower ratings. 
        This suggests that delayed deliveries may be contributing to negative customer experiences. 
        Addressing shipping efficiency and ensuring timely deliveries could help improve customer satisfaction and potentially lead to better review scores.
        """)
        
        fig7b = px.bar(x=avg_price['review_score'], y=avg_price['price'], title="Avg. Price")
        fig7b = fig7b.update_layout(xaxis_title="Review Score", yaxis_title="Price")
        st.plotly_chart(fig7b)        
        st.markdown("""        
        The analysis reveals that average prices do not have a linear relationship with review scores; 
        however, lower ratings tend to correspond with slightly higher prices. 
        While there appears to be some correlation between price and rating, it is not as significant as the relationship observed with shipping time. 
        This insight suggests that while price may influence customer satisfaction to some extent, factors like delivery experience play a more critical role in shaping reviews.
        """)

        shipping_fee = pd.read_csv('data/freight_value.csv')
        fig8 = px.bar(x=shipping_fee['review_score'], y=shipping_fee['freight_value'], title="Impact of Shipping Fees on Ratings")
        fig8 = fig8.update_layout(xaxis_title="Review Score", yaxis_title="Freight Value")
        st.plotly_chart(fig8)
        st.markdown("""
        The data shows a clear linear trend between higher shipping fees and lower review scores, indicating a strong negative correlation. 
        Customers who pay more for shipping tend to leave lower ratings, suggesting dissatisfaction with the cost relative to the perceived value. 
        Reducing shipping fees or improving service quality could help enhance customer satisfaction and improve overall ratings.
        """)
        
        fig9 = px.box(health_vs_hygiene, 
                     x='product_category_name_english', y='review_score', title="Review Scores by Category")
        fig9 = fig9.update_layout(xaxis_title="Product category", yaxis_title="Review Score")

        st.plotly_chart(fig9)
        st.markdown("""
        Review scores vary significantly by category, with different customer expectations influencing ratings. 
        For example, a 3-star review in the **Diapers & Hygiene** category would be above the median, while the same rating in **Health & Beauty** would be among the lowest. 
        This variation suggests that customer satisfaction is highly context-dependent, and category-specific benchmarks should be considered when evaluating product performance and customer feedback.
        """)

        fig10 = px.bar(photos, 
                      x='review_score', y='product_photos_qty', title="Impact of Product Photos Quantity on Reviews")
        fig10 = fig10.update_layout(xaxis_title="Review Score", yaxis_title="Avg. Photo Quantity")
        st.plotly_chart(fig10)
        st.markdown("""
        The data indicates that the number of product photos has minimal impact on review scores. 
        This suggests that factors such as product quality, pricing, and delivery experience play a more significant role in customer satisfaction. 
        While high-quality images are important for conversions, other elements may have a stronger influence on overall ratings.
        """)

if page == "Conclusions":

    st.markdown("""
        ### **Conclusion & Key Recommendations**  

        Our analysis of e-commerce review data has provided valuable insights into customer behavior, sales trends, and the factors influencing review scores. We identified that shopping activity peaks in the afternoons, particularly on Mondays, and that sellers on our platform can achieve significant revenues—up to $100,000 in a single day. However, maintaining high customer satisfaction is essential for sustained success.  

        Customer reviews reveal that **shipping time** is a critical factor affecting ratings, while **price plays a smaller role, provided the product is of high quality**. Additionally, **shipping fees can negatively impact scores**, and **certain product categories receive significantly different ratings**. Interestingly, the **number of product photos does not influence customer satisfaction**, and **products with more reviews tend to have a more stable and higher rating distribution**. Furthermore, we detected anomalies in review patterns, suggesting potential review manipulation or data inconsistencies, which warrant further investigation.  
        ### **Actionable Recommendations**        
    """)
      
    col1, col2 = st.columns(2)

    if col1.button("For Sellers & the e-commerce Platform"):
        st.markdown("""    
            #### **For Sellers:**  
            ✅ **Optimize Shipping & Reduce Delays**: Prioritize fast and reliable delivery, as slow shipping is one of the primary causes of negative reviews.  
            ✅ **Adjust Pricing Strategically**: Higher-priced items can still receive good ratings if they offer strong quality and value. Consider emphasizing quality over just lowering prices.  
            ✅ **Monitor Shipping Fees**: Excessive shipping costs can lead to dissatisfaction. Where possible, offer competitive or free shipping options to improve ratings.  
            ✅ **Encourage More Reviews**: Products with more than 20 reviews tend to have more consistent and higher ratings. Encourage customers to leave feedback to improve credibility and visibility.  
            ✅ **Time Promotions Effectively**: Since Monday afternoons are peak shopping times, schedule discounts and special offers during these periods to maximize sales.  

            #### **For Us as an E-commerce Platform:**  
            📌 **Attract More Sellers with Proven Revenue Potential**: Highlight data showing that sellers can generate up to $100,000 in a single day to attract new merchants.  
            📌 **Enhance Trust & Combat Review Fraud**: Investigate irregular review patterns to prevent potential score manipulation and maintain credibility.  
            📌 **Improve Logistics & Offer Faster Shipping Solutions**: Partner with logistics providers to reduce shipping times and optimize delivery networks.  
            📌 **Provide Insights to Sellers**: Offer dashboards showing shipping performance, pricing trends, and customer feedback to help sellers improve their ratings.  
            📌 **Implement Category-Specific Review Analysis**: Since different product categories receive varying ratings, develop tailored strategies to help sellers improve their category-specific performance.  

            By implementing these strategies, we can create a more efficient, customer-friendly, and high-performing e-commerce ecosystem that benefits both sellers and shoppers. 🚀
                    
         """)      

    if col2.button("Data Analysis Team Next Steps"):
        st.markdown("""
            ### **Actionable Recommendations for the Data Analysis Team**  

            As data analysts, our role is crucial in ensuring that insights from customer reviews and sales trends drive informed decision-making. Based on our findings, we can take several key steps to improve data accuracy, uncover deeper insights, and enhance our e-commerce platform’s performance.  

            #### **1️⃣ Improve Data Quality & Integrity**  
            ✅ **Investigate Review Anomalies**: Analyze irregular review patterns (e.g., concentrated review dates, sudden rating spikes) to detect potential fraud or manipulation. Implement anomaly detection models to flag suspicious activity.  
            ✅ **Validate Data Collection Processes**: Ensure there are no errors in data extraction, especially when identifying unusual patterns such as the blue product’s review clustering.  
            ✅ **Monitor Class Imbalance in Reviews**: Since there is a noticeable imbalance in review scores, apply resampling techniques or weighting methods when building predictive models.  

            #### **2️⃣ Enhance Review-Based Insights**  
            ✅ **Develop a Review Impact Model**: Build a predictive model to quantify how much different factors (e.g., shipping time, price, fees) contribute to a product’s final rating.  
            ✅ **Create a Sentiment Analysis Dashboard**: Implement NLP techniques to analyze review text and uncover hidden trends in customer sentiment beyond numerical ratings.  
            ✅ **Segment Review Analysis by Category**: Since ratings vary significantly across product categories, develop category-specific insights to provide tailored recommendations to sellers.  

            #### **3️⃣ Optimize Seller & Customer Experience**  
            ✅ **Track Seller Performance Metrics**: Build a dashboard to monitor key seller performance indicators, such as average review scores, shipping times, and return rates, helping sellers improve their service.  
            ✅ **Identify High-Risk Products Early**: Use review distribution patterns to flag products likely to receive poor ratings and proactively notify sellers.  
            ✅ **Study the Impact of Promotions on Reviews & Sales**: Since sales peak on Monday afternoons, analyze how promotions influence review scores and long-term customer retention.  

            #### **4️⃣ Improve Logistics & Pricing Strategies**  
            ✅ **Analyze Shipping Delays & Their Causes**: Investigate geographic or operational factors that contribute to long shipping times and propose solutions for optimization.  
            ✅ **Evaluate the Price-Quality Relationship**: Conduct deeper pricing analysis to determine the optimal price range for maximizing both revenue and customer satisfaction.  
            ✅ **Assess the Impact of Shipping Fees**: Develop pricing models to predict how changes in shipping costs impact overall sales and review scores.  

            By implementing these action items, we can drive more precise, data-driven decision-making for our e-commerce platform, improving both the seller and customer experience. 🚀                    
        """)