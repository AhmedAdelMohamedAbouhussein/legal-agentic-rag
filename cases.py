"""
cases.py — 10 pre-loaded legal case scenarios from the capstone spec.
Each case has: title, category, facts, legal_issues, rag_query.
"""

LEGAL_CASES = [
    {
        "id": 1,
        "title": "Force Majeure in Supply Chain (COVID-19)",
        "category": "contracts",
        "facts": (
            "Alpha Corp (buyer) and BetaSupply (supplier) signed a 3-year supply agreement "
            "in 2018. In 2020, BetaSupply failed to deliver components due to factory shutdowns "
            "caused by COVID-19. The contract includes a force majeure clause listing "
            "'natural disasters, acts of God, and governmental action' but does not mention pandemics."
        ),
        "legal_issues": [
            "Does COVID-19 qualify as force majeure under this clause?",
            "Does the doctrine of frustration of purpose apply?",
            "What damages, if any, can Alpha Corp claim?",
        ],
        "rag_query": (
            "Does COVID-19 qualify as force majeure under a supply contract clause "
            "listing natural disasters and governmental action?"
        ),
    },
    {
        "id": 2,
        "title": "Non-Compete Clause Enforceability (California)",
        "category": "contracts",
        "facts": (
            "A software engineer in California signed an employment agreement with a 2-year, "
            "nationwide non-compete clause preventing work for any competitor in the SaaS industry. "
            "She resigned and joined a competitor in Texas. The original employer seeks an injunction."
        ),
        "legal_issues": [
            "Is the non-compete enforceable under California law (Bus & Prof Code §16600)?",
            "Does Texas choice-of-law provision override California's public policy?",
            "Can the employer seek injunctive relief?",
        ],
        "rag_query": (
            "Enforceability of non-compete clauses under California Business and "
            "Professions Code Section 16600 for tech employees"
        ),
    },
    {
        "id": 3,
        "title": "GDPR Data Breach Liability",
        "category": "both",
        "facts": (
            "A European fintech company suffered a data breach exposing 500,000 customer records. "
            "Their cloud vendor processed the data under a data processing agreement (DPA). "
            "The DPA states the vendor implements 'industry-standard security measures' but does "
            "not specify encryption requirements. Customers are filing a class action."
        ),
        "legal_issues": [
            "Is the vendor a data processor or joint controller under GDPR?",
            "Does 'industry-standard security' satisfy Article 32 technical safeguards?",
            "What is the maximum fine under Article 83?",
            "Can liability be limited by the DPA's liability cap?",
        ],
        "rag_query": (
            "GDPR Article 32 security obligations for cloud data processors and "
            "liability under data processing agreements"
        ),
    },
    {
        "id": 4,
        "title": "Intellectual Property Ownership in Employment",
        "category": "contracts",
        "facts": (
            "A data scientist employed full-time developed a machine learning algorithm during "
            "evenings on her personal laptop, using her own resources. The employer claims ownership "
            "under a standard IP assignment clause that assigns 'all inventions conceived during employment.'"
        ),
        "legal_issues": [
            "Does the employer's IP clause cover inventions developed on personal time?",
            "Do California Labor Code §§2870-2872 carve-out protections apply?",
            "Is the IP clause overbroad and potentially unenforceable?",
        ],
        "rag_query": (
            "Employee IP assignment clause enforceability for inventions developed on "
            "personal time under California Labor Code 2870"
        ),
    },
    {
        "id": 5,
        "title": "Liquidated Damages vs. Penalty Clauses",
        "category": "both",
        "facts": (
            "An e-commerce platform contracted a logistics company for $50/package/day delay penalty. "
            "The logistics company was 5 days late on 8,000 packages, resulting in a $2,000,000 claim."
        ),
        "legal_issues": [
            "Is $50/package/day a genuine pre-estimate of loss or an unenforceable penalty?",
            "What was the actual loss to the platform?",
            "How do English courts (Cavendish Square) vs US courts assess this?",
        ],
        "rag_query": (
            "Liquidated damages clause enforceability test genuine pre-estimate of loss "
            "versus unenforceable penalty clause"
        ),
    },
    {
        "id": 6,
        "title": "SaaS Auto-Renewal Dispute",
        "category": "contracts",
        "facts": (
            "A startup was automatically charged $120,000 for a third year of a SaaS enterprise "
            "subscription. The auto-renewal clause required written cancellation 90 days before "
            "renewal and was buried in Section 12.4 of a 40-page contract signed electronically."
        ),
        "legal_issues": [
            "Is the auto-renewal clause enforceable if buried in the contract?",
            "Does the electronic signature satisfy Statute of Frauds?",
            "Do B2B contract laws require conspicuous notice of auto-renewal?",
        ],
        "rag_query": (
            "Auto-renewal clause enforceability conspicuous notice requirement SaaS "
            "enterprise contracts electronic signature"
        ),
    },
    {
        "id": 7,
        "title": "Wrongful Termination & Whistleblower Retaliation",
        "category": "case_law",
        "facts": (
            "An employee was terminated after raising safety concerns about a product defect to the CEO. "
            "The company claims termination was for performance reasons. The employee alleges the PIPs "
            "were fabricated and the real reason was retaliation for whistleblowing."
        ),
        "legal_issues": [
            "Does the employee have a whistleblower retaliation claim?",
            "Do Sarbanes-Oxley or Dodd-Frank whistleblower protections apply?",
            "Is the at-will employment doctrine overcome?",
        ],
        "rag_query": (
            "Whistleblower retaliation wrongful termination at-will employment "
            "exceptions Sarbanes-Oxley private company"
        ),
    },
    {
        "id": 8,
        "title": "Construction Contract Scope Creep",
        "category": "contracts",
        "facts": (
            "A general contractor signed a fixed-price construction contract for $5M. During construction, "
            "the owner requested 23 additional change orders valued at $1.2M. The contractor completed "
            "all changes but the owner refuses to pay, arguing they were within the original scope."
        ),
        "legal_issues": [
            "What constitutes a valid change order under AIA contract terms?",
            "Does verbal instruction from the owner constitute a binding change order?",
            "Can the contractor claim unjust enrichment for completed but disputed work?",
        ],
        "rag_query": (
            "Construction contract change order validity verbal instructions "
            "scope of work dispute unjust enrichment"
        ),
    },
    {
        "id": 9,
        "title": "Trade Secret Misappropriation by Departing Employee",
        "category": "both",
        "facts": (
            "A departing VP of Sales emailed himself a list of 5,000 enterprise customer contacts, "
            "pricing structures, and deal pipeline information before leaving. He joined a direct "
            "competitor and used this data to approach the former employer's clients."
        ),
        "legal_issues": [
            "Does the customer list qualify as a trade secret under the DTSA?",
            "Is the NDA's confidentiality clause broad enough to cover this data?",
            "What remedies are available — injunction, damages, attorney's fees?",
        ],
        "rag_query": (
            "Trade secret misappropriation customer list Defend Trade Secrets Act "
            "injunctive relief departing employee"
        ),
    },
    {
        "id": 10,
        "title": "Arbitration Clause Unconscionability",
        "category": "case_law",
        "facts": (
            "A consumer signed an online ToS with a mandatory arbitration clause and class action waiver. "
            "The consumer is part of a class alleging systematic overcharging of $15/month over 3 years. "
            "Individual arbitration would cost more than any individual's recovery."
        ),
        "legal_issues": [
            "Is the arbitration clause unconscionable where individual recovery < arbitration costs?",
            "Does AT&T Mobility v. Concepcion preclude unconscionability challenges?",
            "Is there a public policy exception for small-value claims?",
        ],
        "rag_query": (
            "Mandatory arbitration clause class action waiver unconscionability "
            "consumer contracts small value claims"
        ),
    },
]
