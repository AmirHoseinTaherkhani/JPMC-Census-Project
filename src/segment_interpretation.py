"""
Customer Segment Interpretation

Proper segment personas based on statistical profiles from clustering analysis.
Manually curated to reflect actual segment differences rather than population modes.

Author: Senior Data Scientist, JP Morgan
Date: November 2025
"""

# Based on segmentation_results.json segment_profiles

SEGMENT_INTERPRETATIONS = {
    "Segment 0: Working Professionals": {
        "size": "37,948 (19.3%)",
        "high_income_rate": "10.9%",
        "key_characteristics": [
            "Average work engagement (97% of normal weeks worked)",
            "Moderate capital gains ownership (6.8%)",
            "Moderate dividend income (7.1%)",
            "Core middle-income segment"
        ],
        "persona_description": "Employed individuals with stable jobs and some investment activity. Represent the working middle class with moderate income potential. Likely skilled workers, junior professionals, or experienced non-managerial employees.",
        "marketing_strategy": "Balanced approach focusing on value and quality. Appeal to career advancement aspirations while remaining price-conscious. Products that improve work-life balance and long-term financial stability.",
        "recommended_channels": [
            "LinkedIn and professional networks",
            "Email marketing",
            "Business publications",
            "Workplace benefit programs"
        ],
        "messaging_themes": [
            "Career advancement",
            "Financial security",
            "Quality and reliability",
            "Time-saving solutions"
        ],
        "budget_allocation": "Medium (15-20% of marketing budget)"
    },
    
    "Segment 1: Low-Income/Unemployed": {
        "size": "60,850 (31.0%)",
        "high_income_rate": "0.03%",
        "key_characteristics": [
            "Very low work engagement (86% below average weeks worked)",
            "Minimal capital gains (0.2%)",
            "Minimal dividend income (0.3%)",
            "Largest segment but lowest income"
        ],
        "persona_description": "Unemployed, students, part-time workers, or those not in the labor force. Includes young people early in careers, individuals between jobs, and those with limited work history. Minimal discretionary income and investment activity.",
        "marketing_strategy": "Volume-based approach with cost-effective channels. Focus on accessibility, affordability, and future potential. Entry-level products and services that don't require significant disposable income.",
        "recommended_channels": [
            "Social media platforms",
            "Digital advertising",
            "Community outreach",
            "Student programs and partnerships"
        ],
        "messaging_themes": [
            "Affordability and value",
            "Accessibility",
            "Future opportunity",
            "Getting started/entry-level"
        ],
        "budget_allocation": "Low (5-10% of marketing budget despite size)"
    },
    
    "Segment 2: Affluent Professionals": {
        "size": "3,802 (1.9%)",
        "high_income_rate": "30.5%",
        "key_characteristics": [
            "Moderate-high work engagement (74% of normal weeks)",
            "NO capital gains (0.0%) - interesting anomaly",
            "High dividend income (15.8%) - passive investors",
            "Premium segment with highest income rate"
        ],
        "persona_description": "High-income professionals with substantial dividend income but no capital gains, suggesting long-term buy-and-hold investors or inherited wealth. Despite moderate work hours, they maintain high income through investments. Likely established professionals, business owners, or those with significant passive income streams.",
        "marketing_strategy": "Premium, high-touch relationship building. Focus on exclusive access, wealth management, and legacy planning. Products emphasizing quality, status, and long-term value over price.",
        "recommended_channels": [
            "Private banking and wealth management",
            "Financial advisor networks",
            "Premium publications",
            "Exclusive events and experiences"
        ],
        "messaging_themes": [
            "Wealth preservation",
            "Exclusive access",
            "Premium quality and service",
            "Legacy and long-term planning"
        ],
        "budget_allocation": "High (30-35% of marketing budget for this small but valuable segment)"
    },
    
    "Segment 3: Retirees/Older Adults": {
        "size": "42,920 (21.9%)",
        "high_income_rate": "1.8%",
        "key_characteristics": [
            "Low work engagement (84% below average weeks)",
            "Some capital gains activity (3.5%)",
            "Moderate dividend income (10.5%)",
            "Older demographic with modest retirement income"
        ],
        "persona_description": "Likely retirees or older individuals with reduced work engagement but moderate investment portfolios. Living on social security, pensions, and modest investment income. Have accumulated some assets over lifetime but not high current income.",
        "marketing_strategy": "Traditional marketing with focus on trust, reliability, and community. Products serving retiree needs - healthcare, leisure, downsizing, fixed-income optimization. Emphasize stability and proven track record.",
        "recommended_channels": [
            "Traditional media (TV, print)",
            "Direct mail",
            "Community centers and senior programs",
            "Local events and sponsorships"
        ],
        "messaging_themes": [
            "Trust and reliability",
            "Community and belonging",
            "Health and wellness",
            "Simplicity and ease of use"
        ],
        "budget_allocation": "Low-Medium (10-15% of marketing budget)"
    },
    
    "Segment 4: Established Professionals": {
        "size": "50,774 (25.9%)",
        "high_income_rate": "12.4%",
        "key_characteristics": [
            "High work engagement (96% of normal weeks worked)",
            "Moderate capital gains (6.2%)",
            "Moderate dividend income (8.1%)",
            "Largest core income segment"
        ],
        "persona_description": "Fully employed professionals with stable careers and growing investment portfolios. Represent the upper-middle class with consistent work engagement and diversified investments. Likely mid-career professionals, managers, skilled tradespeople with solid financial footing.",
        "marketing_strategy": "Balanced premium approach emphasizing both value and quality. Appeal to career success and financial growth aspirations. Products supporting wealth building, career advancement, and lifestyle enhancement.",
        "recommended_channels": [
            "LinkedIn and professional networks",
            "Email campaigns",
            "Industry conferences and events",
            "Business and financial media"
        ],
        "messaging_themes": [
            "Success and achievement",
            "Wealth building",
            "Premium quality worth the investment",
            "Work-life balance"
        ],
        "budget_allocation": "Medium-High (25-30% of marketing budget)"
    }
}


# Summary table for quick reference
SEGMENT_SUMMARY_TABLE = """
+----------+---------------------------+--------+-------------+------------------+
| Segment  | Name                      | Size   | High Income | Value Tier       |
+----------+---------------------------+--------+-------------+------------------+
| 0        | Working Professionals     | 19.3%  | 10.9%       | Core             |
| 1        | Low-Income/Unemployed     | 31.0%  | 0.03%       | Growth           |
| 2        | Affluent Professionals    | 1.9%   | 30.5%       | Premium          |
| 3        | Retirees/Older Adults     | 21.9%  | 1.8%        | Growth           |
| 4        | Established Professionals | 25.9%  | 12.4%       | Core             |
+----------+---------------------------+--------+-------------+------------------+

Key Insights:
- Segment 2 (Affluent Professionals) is small but extremely valuable (30.5% high-income rate)
- Segments 1 & 3 comprise 53% of population but <2% high-income rates
- Segments 0 & 4 are the core middle class (45% of population, 11-12% high-income rates)
- Clear income stratification enables targeted marketing strategies
"""

# Marketing budget allocation recommendation
BUDGET_ALLOCATION = {
    "Segment 0": "15-20%",
    "Segment 1": "5-10%",
    "Segment 2": "30-35%",  # Small segment but highest ROI
    "Segment 3": "10-15%",
    "Segment 4": "25-30%"
}

if __name__ == "__main__":
    print("Customer Segment Interpretations")
    print("=" * 80)
    print(SEGMENT_SUMMARY_TABLE)
    print("\nDetailed segment profiles available in SEGMENT_INTERPRETATIONS dictionary")