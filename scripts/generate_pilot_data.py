"""
Generate realistic synthetic text documents for pipeline validation.
Creates 20 samples per class (invoice, receipt, email, contract) as .txt files.
These are pilot samples to validate the end-to-end pipeline before real data is added.
"""

import os
import random
import string

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJ_ROOT, "data", "raw")

random.seed(42)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rand_name():
    first = random.choice(["James", "Sarah", "Michael", "Emily", "David", "Anna",
                           "Robert", "Laura", "Thomas", "Karen", "Daniel", "Sophie"])
    last = random.choice(["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                          "Miller", "Davis", "Martinez", "Anderson", "Taylor", "Wilson"])
    return f"{first} {last}"

def _rand_company():
    prefix = random.choice(["Acme", "Global", "Pacific", "Northern", "Summit",
                            "Apex", "Prime", "Atlas", "Sterling", "Vertex"])
    suffix = random.choice(["Corp", "Ltd", "Inc", "Solutions", "Industries",
                            "Group", "Holdings", "Technologies", "Services", "Partners"])
    return f"{prefix} {suffix}"

def _rand_date():
    y = random.choice([2023, 2024, 2025])
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"

def _rand_amount():
    return round(random.uniform(50, 25000), 2)

def _rand_email_addr(name=None):
    if name:
        local = name.lower().replace(" ", ".") + str(random.randint(1, 99))
    else:
        local = "".join(random.choices(string.ascii_lowercase, k=8))
    domain = random.choice(["gmail.com", "outlook.com", "company.org",
                            "business.net", "corporate.io"])
    return f"{local}@{domain}"

# ---------------------------------------------------------------------------
# Document generators
# ---------------------------------------------------------------------------

def generate_invoice(idx):
    inv_num = f"INV-{random.randint(2024, 2025)}-{random.randint(1000, 9999)}"
    issuer = _rand_company()
    recipient = _rand_company()
    inv_date = _rand_date()
    due_date = _rand_date()
    items = []
    n_items = random.randint(1, 6)
    subtotal = 0
    for _ in range(n_items):
        desc = random.choice(["Consulting Services", "Software License",
                              "Hardware Equipment", "Maintenance Fee",
                              "Training Session", "Project Management",
                              "Cloud Hosting", "Data Analytics Service",
                              "Annual Subscription", "Technical Support"])
        qty = random.randint(1, 10)
        unit_price = round(random.uniform(50, 3000), 2)
        line_total = round(qty * unit_price, 2)
        subtotal += line_total
        items.append(f"  {desc:<30s}  Qty: {qty:>3d}  Unit Price: ${unit_price:>10.2f}  Total: ${line_total:>10.2f}")

    tax = round(subtotal * random.choice([0.06, 0.08, 0.10, 0.21]), 2)
    total = round(subtotal + tax, 2)

    text = f"""INVOICE

Invoice Number: {inv_num}
Invoice Date: {inv_date}
Due Date: {due_date}

Bill From:
  {issuer}
  {random.randint(1,999)} {random.choice(['Main St', 'Oak Ave', 'Park Rd', 'Business Blvd'])}
  {random.choice(['New York', 'London', 'Amsterdam', 'Berlin', 'Paris'])}, {random.randint(10000,99999)}

Bill To:
  {recipient}
  {random.randint(1,999)} {random.choice(['Elm St', 'River Rd', 'High St', 'Commerce Dr'])}
  {random.choice(['Chicago', 'Sydney', 'Toronto', 'Madrid', 'Tokyo'])}, {random.randint(10000,99999)}

{'='*80}
ITEMS
{'='*80}
{chr(10).join(items)}

{'='*80}
  Subtotal:   ${subtotal:>10.2f}
  Tax:        ${tax:>10.2f}
  TOTAL DUE:  ${total:>10.2f}
{'='*80}

Payment Terms: Net 30
Payment Method: Bank Transfer

Thank you for your business.
"""
    return text


def generate_receipt(idx):
    store = random.choice(["QuickMart", "FreshGrocer", "TechZone", "CoffeeHouse",
                           "BookNest", "PharmaPlus", "GreenMarket", "StyleHub",
                           "GadgetWorld", "PetCorner"])
    date = _rand_date()
    time = f"{random.randint(8,21):02d}:{random.randint(0,59):02d}"
    items = []
    total = 0
    n_items = random.randint(2, 8)
    for _ in range(n_items):
        name = random.choice(["Coffee Latte", "Whole Milk 1L", "Bread Loaf",
                              "Chicken Breast", "Organic Eggs", "USB Cable",
                              "Notebook A5", "Shampoo 500ml", "Orange Juice",
                              "Banana 1kg", "Rice 2kg", "Dark Chocolate",
                              "Paper Towels", "Toothpaste", "Cereal Box"])
        price = round(random.uniform(1.5, 45.0), 2)
        qty = random.randint(1, 3)
        line = round(price * qty, 2)
        total += line
        items.append(f"  {name:<25s} x{qty}  ${line:>7.2f}")

    tax = round(total * random.choice([0.05, 0.07, 0.10]), 2)
    grand = round(total + tax, 2)
    paid = random.choice(["VISA ****4532", "MASTERCARD ****8821",
                          "CASH", "AMEX ****1190", "DEBIT ****6677"])

    text = f"""{'='*40}
        {store}
  {random.randint(1,999)} {random.choice(['Main St', 'High St', 'Market Sq'])}
  Tel: {random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}
{'='*40}
  Date: {date}  Time: {time}
  Cashier: #{random.randint(1,20)}
{'='*40}

{chr(10).join(items)}

{'-'*40}
  Subtotal:        ${total:>7.2f}
  Tax:             ${tax:>7.2f}
  TOTAL:           ${grand:>7.2f}
{'-'*40}
  Payment: {paid}
{'='*40}

  Thank you for shopping at {store}!
  Receipt #{random.randint(100000,999999)}
"""
    return text


def generate_email(idx):
    sender = _rand_name()
    recipient = _rand_name()
    subject_templates = [
        "Re: Meeting rescheduled to next week",
        "Follow-up on Q{q} budget proposal",
        "Project update: milestone {m} completed",
        "Action required: review attached document",
        "Invitation: team offsite on {d}",
        "Quick question about the deployment",
        "FYI: policy change effective {d}",
        "Re: Re: Client feedback summary",
        "Reminder: submission deadline approaching",
        "Introduction: new team member joining",
        "Agenda for tomorrow's standup",
        "Request for time off: {d} to {d2}",
        "Updated: sprint retrospective notes",
        "Clarification needed on requirements",
    ]
    subj = random.choice(subject_templates).format(
        q=random.randint(1, 4), m=random.randint(1, 5),
        d=_rand_date(), d2=_rand_date())

    bodies = [
        f"Hi {recipient.split()[0]},\n\nJust wanted to follow up on our conversation from last week regarding the project timeline. "
        f"I've reviewed the updated schedule and I think we can move forward with the proposed changes. "
        f"Could you please confirm that the budget allocation for Q{random.randint(1,4)} has been approved?\n\n"
        f"Also, I'd like to schedule a brief call to discuss the client's feedback on the latest deliverable. "
        f"Let me know your availability this week.\n\nBest regards,\n{sender}",

        f"Dear {recipient.split()[0]},\n\nThank you for sending over the report. I've gone through the key findings "
        f"and have a few comments:\n\n1. The market analysis section needs more recent data\n"
        f"2. Please update the financial projections for the next quarter\n"
        f"3. The executive summary looks good overall\n\n"
        f"Can we set up a meeting to go over these points? I'm available on {_rand_date()} afternoon.\n\n"
        f"Thanks,\n{sender}",

        f"Hello team,\n\nThis is a reminder that the deadline for submitting your progress reports is {_rand_date()}. "
        f"Please ensure all documents are uploaded to the shared drive by end of day.\n\n"
        f"Key items to include:\n- Current project status\n- Blockers and risks\n- Next steps and milestones\n\n"
        f"If you have any questions, don't hesitate to reach out.\n\nRegards,\n{sender}",

        f"Hi {recipient.split()[0]},\n\nI wanted to let you know that we've completed milestone {random.randint(1,5)} "
        f"of the project. The development team has finished the integration testing and all critical tests are passing.\n\n"
        f"We're now moving into the user acceptance testing phase. The QA team will need access to the staging environment "
        f"by {_rand_date()}.\n\nPlease coordinate with IT to ensure the environment is ready.\n\n"
        f"Cheers,\n{sender}",

        f"Dear {recipient.split()[0]},\n\nPlease find attached the meeting minutes from today's session. "
        f"The key decisions were:\n\n- Approved the revised project timeline\n- Budget increase of ${_rand_amount():.2f} for Q{random.randint(1,4)}\n"
        f"- New hire to start on {_rand_date()}\n- Next review meeting scheduled for {_rand_date()}\n\n"
        f"Let me know if I missed anything.\n\nBest,\n{sender}",
    ]

    body = random.choice(bodies)

    text = f"""From: {sender} <{_rand_email_addr(sender)}>
To: {recipient} <{_rand_email_addr(recipient)}>
Date: {_rand_date()}
Subject: {subj}

{body}
"""
    return text


def generate_contract(idx):
    party_a = _rand_company()
    party_b = _rand_company()
    effective_date = _rand_date()
    contract_types = ["Service Agreement", "Non-Disclosure Agreement",
                      "Employment Contract", "Consulting Agreement",
                      "Software License Agreement", "Partnership Agreement"]
    ctype = random.choice(contract_types)

    clauses = [
        f"""1. DEFINITIONS
1.1 "Agreement" means this {ctype} between the parties.
1.2 "Effective Date" means {effective_date}.
1.3 "Confidential Information" means any non-public information disclosed by either party to the other.""",

        f"""2. SCOPE OF SERVICES
2.1 {party_a} agrees to provide the services described in Exhibit A attached hereto.
2.2 The services shall be performed in a professional and workmanlike manner consistent with industry standards.
2.3 {party_b} shall provide reasonable access and cooperation as necessary for the performance of the services.""",

        f"""3. TERM AND TERMINATION
3.1 This Agreement shall commence on the Effective Date and continue for a period of {random.choice([12, 24, 36])} months.
3.2 Either party may terminate this Agreement upon {random.choice([30, 60, 90])} days written notice.
3.3 Upon termination, all obligations under this Agreement shall cease except those which by their nature survive termination.""",

        f"""4. COMPENSATION AND PAYMENT
4.1 {party_b} shall pay {party_a} a total fee of ${_rand_amount():,.2f} for the services rendered.
4.2 Payment shall be made within {random.choice([15, 30, 45])} days of receipt of invoice.
4.3 Late payments shall bear interest at a rate of {random.choice([1.0, 1.5, 2.0])}% per month.""",

        f"""5. CONFIDENTIALITY
5.1 Each party agrees to hold in confidence all Confidential Information received from the other party.
5.2 Confidential Information shall not be disclosed to any third party without prior written consent.
5.3 This obligation of confidentiality shall survive the termination of this Agreement for a period of {random.choice([2, 3, 5])} years.""",

        f"""6. INTELLECTUAL PROPERTY
6.1 All intellectual property created in the course of performing services shall belong to {party_b}.
6.2 {party_a} retains ownership of any pre-existing intellectual property.
6.3 {party_a} grants {party_b} a non-exclusive license to use any pre-existing intellectual property incorporated into the deliverables.""",

        f"""7. LIABILITY AND INDEMNIFICATION
7.1 Neither party shall be liable for any indirect, incidental, or consequential damages.
7.2 The total liability of either party shall not exceed the total fees paid under this Agreement.
7.3 Each party agrees to indemnify the other against claims arising from its breach of this Agreement.""",

        f"""8. GOVERNING LAW
8.1 This Agreement shall be governed by the laws of {random.choice(['the State of New York', 'England and Wales', 'the Netherlands', 'the State of California'])}.
8.2 Any disputes shall be resolved through {random.choice(['binding arbitration', 'mediation', 'the courts of competent jurisdiction'])}.""",

        f"""9. GENERAL PROVISIONS
9.1 This Agreement constitutes the entire agreement between the parties.
9.2 No amendment or modification shall be valid unless made in writing and signed by both parties.
9.3 If any provision is found to be unenforceable, the remaining provisions shall continue in full force and effect.""",
    ]

    n_clauses = random.randint(5, len(clauses))
    selected = random.sample(clauses, n_clauses)

    text = f"""{ctype.upper()}

This {ctype} ("Agreement") is entered into as of {effective_date} by and between:

Party A: {party_a}, a company organized and existing under the laws of {random.choice(['Delaware', 'England', 'the Netherlands', 'Ontario'])}, with its principal office at {random.randint(1,999)} {random.choice(['Corporate Blvd', 'Legal Ave', 'Commerce St'])} ("Provider")

and

Party B: {party_b}, a company organized and existing under the laws of {random.choice(['California', 'Scotland', 'Germany', 'British Columbia'])}, with its principal office at {random.randint(1,999)} {random.choice(['Enterprise Dr', 'Business Park', 'Innovation Way'])} ("Client")

WHEREAS, {party_a} possesses expertise in {random.choice(['software development', 'consulting services', 'technology solutions', 'business analytics', 'project management'])}; and

WHEREAS, {party_b} desires to engage {party_a} to provide certain services;

NOW, THEREFORE, in consideration of the mutual covenants and agreements herein contained, the parties agree as follows:

{chr(10).join(selected)}

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

______________________________          ______________________________
{party_a}                               {party_b}
Authorized Representative               Authorized Representative
Date: {effective_date}                  Date: {effective_date}
"""
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = {
    "invoice": generate_invoice,
    "receipt": generate_receipt,
    "email": generate_email,
    "contract": generate_contract,
}

def main():
    n_per_class = 20
    total = 0
    for label, gen_fn in GENERATORS.items():
        class_dir = os.path.join(RAW_DIR, label)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(1, n_per_class + 1):
            text = gen_fn(i)
            fname = f"{label}_{i:03d}.txt"
            fpath = os.path.join(class_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)
            total += 1
        print(f"  [{label}] generated {n_per_class} samples in {class_dir}")

    print(f"\nTotal pilot documents generated: {total}")


if __name__ == "__main__":
    main()
