from datetime import datetime

def compute_posting_frequency(posts):
    dates = [datetime.strptime(p['timestamp'], "%Y-%m-%d") for p in posts]
    total_days = (max(dates) - min(dates)).days + 1
    weeks = total_days / 7
    return len(posts) / weeks if weeks else len(posts)
def average_views(posts):
    return sum(p['views'] for p in posts) / len(posts)
def top_post(posts):
    top = max(posts, key=lambda p: p['views'])
    return {
        "product": top.get("product_name"),
        "price": top.get("price"),
        "views": top["views"]
    }
def average_price(posts):
    prices = [p["price"] for p in posts if p.get("price")]
    return sum(prices) / len(prices) if prices else 0
def lending_score(avg_views, posts_per_week, avg_price):
    return (avg_views * 0.5) + (posts_per_week * 0.3) + (avg_price * 0.2)
