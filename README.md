# Project Proposal: RiskShield Booking AI
### *Predictive Revenue Protection & Smart Deposit Engine for Hospitality Operations*

**Prepared for:** Executive Leadership & Revenue Management Committee, Aurelia Hospitality Group  
**Prepared by:** Data Science & Commercial Analytics Team  
**Document Type:** Project Initiation Proposal (Backed by Working ML Proof-of-Concept)  
**Target Properties:** Urban Business ("City Hotel") & Leisure ("Resort Hotel") Portfolio  

---

## 1. Executive Summary

Every night a hotel room sits empty, its revenue is lost permanently. Unlike physical retail goods, unsold hotel inventory cannot be stored and resold tomorrow. Across our City and Resort properties, **more than $37\%$ of confirmed reservations are canceled prior to check-in**—creating severe last-minute occupancy gaps, unpredictable staffing schedules, and millions of Euros in lost annual room revenue. Current responses—either absorbing empty rooms or enforcing blanket non-refundable policies—force an unacceptable trade-off between bleeding revenue and driving loyal guests to competitors.

To resolve this, we propose **RiskShield Booking AI**, a dual-model Machine Learning revenue protection engine designed to score every incoming reservation at the exact moment of booking. Using only eight verifiable booking-time attributes (such as lead time, stay length, room category, booking channel, and special requests), RiskShield Booking AI pairs a **Dynamic Price Valuation Model (Regression)** with a **Cancellation Risk Model (Classification)** to quantify exact **Revenue at Risk** in Euros. Instead of penalizing every traveler, the system automatically flags only high-risk, high-value bookings for targeted credit-card holds or pre-arrival confirmation—while approving safe bookings with zero friction.

We have already built and validated a **working Proof-of-Concept (PoC)** on **119,390 real historical reservations** (trained strictly on 2015–2016 and tested on unseen 2017 bookings). The prototype confirms that bookings flagged as high-risk cancel at more than double the rate of low-risk bookings and isolates millions of Euros in vulnerable room revenue before check-in day. This proposal requests approval for a **12-week production pilot** on an estimated budget of **€42,500** to connect our validated ML engine to live hotel operations and measure real-world revenue recovery.

---

## 2. Project Background

### The Story on the Front Desk: Why Today's Process Is Failing
Picture a busy August Friday at our Resort property. Three months earlier, every single room was booked solid, so the Revenue Manager closed online inventory and turned away new inquiries. Then, during the final two weeks before arrival, a wave of early Online Travel Agency (OTA) reservations—booked 200+ days in advance with free cancellation—suddenly cancels. Because those cancellations arrive too late to backfill at full rate, high-value suites sit empty while the property still pays fixed housekeeping, utility, and front-desk payroll costs.

Meanwhile, in November, an off-season guest books a single low-rate night at our City property. If the hotel reacts defensively by demanding a strict non-refundable deposit from *every* guest, price-sensitive corporate travelers and returning direct guests abandon checkout and book a competing brand down the street.

### What Is Already Known (Industry Evidence & Benchmarks)
This challenge is well-documented across global hospitality research and our own historical Property Management System (PMS) logs:
* **Scale of the Leakage:** Peer-reviewed hospitality research analyzing 119,390 real hotel reservations (*Antonio, de Almeida, & Nunes, 2019, Data in Brief*) established that overall cancellation rates routinely reach **$37\%$** (exceeding $41\%$ in urban City hotels and $27\%$ in Resort properties).
* **Channel & Lead-Time Dynamics:** Bookings made through Online Travel Agencies (OTAs) with long lead times exhibit sharply elevated cancellation rates, whereas direct bookings and guests who submit specific stay requests (e.g., extra pillows, high floor, baby crib) demonstrate significantly higher arrival commitment.
* **Dynamic Pricing Complexity:** A room's **Average Daily Rate (ADR)** is not fixed; the same room fluctuates dramatically by season, hotel type, room tier, and stay duration.

### Why Past Approaches & Standard Industry Rules Are Insufficient
Hospitality groups have historically relied on three imperfect approaches:
1. **Static Cancellation Rules:** Enforcing universal non-refundable rates across all rooms lowers conversion rates and damages brand loyalty among reliable guests.
2. **Manual Front-Desk Follow-Ups:** Calling every guest before arrival is labor-intensive and unscalable—staff end up spending the same 10 minutes chasing a €65 one-night stay as they do protecting a €1,200 five-night summer suite.
3. **Single-Metric Prediction Models:** Prior academic and vendor attempts typically predict *only* whether a guest will cancel (binary classification). Without pairing cancellation probability with the predicted **financial value** of the booking (`ADR` $\times$ `Nights`), operations teams cannot prioritize which risky bookings actually justify intervention.

### Strategic Relevance to Aurelia Hospitality Group
As distribution costs from third-party OTAs continue to rise, protecting net realized revenue on existing demand is the fastest lever to improve **RevPAR (Revenue Per Available Room)** and operating margin without spending additional marketing capital.

---

## 3. Proposed Solution

### Vision Statement
> *"To equip hotel revenue managers with real-time, booking-level financial visibility—protecting vulnerable room revenue through targeted, data-driven interventions while keeping the booking experience seamless for reliable guests."*

### Solution Architecture (Plain-English Approach)
**RiskShield Booking AI** operates as a lightweight, automated decision layer integrated alongside our existing booking intake workflow. Whenever a new reservation is created, the system evaluates **eight honest booking-time inputs** (`hotel`, `month`, `segment`, `room`, `lead_time`, `nights`, `adults`, `special_requests`)—deliberately excluding any post-check-in records to ensure zero data leakage in live operations—and runs two complementary engines:

1. **Engine 1 — Fair Room Valuation (Regression):** Estimates the fair nightly room rate ($\widehat{\text{ADR}}$) for that stay profile, giving leadership a transparent benchmark of total booking value ($\widehat{\text{ADR}} \times \text{Nights}$).
2. **Engine 2 — Cancellation Risk Scoring (Classification):** Outputs a probability of cancellation ($P_{\text{Cancel}}$) between $0\%$ and $100\%$.
3. **Engine 3 — Automated 3-Tier Revenue Action Policy:** Combines both outputs into pure expected-value business math—requiring **zero arbitrary guesswork**:
   $$\text{Expected Revenue} = (1 - P_{\text{Cancel}}) \times \widehat{\text{ADR}} \times \text{Nights}$$
   $$\text{Revenue at Risk} = P_{\text{Cancel}} \times \widehat{\text{ADR}} \times \text{Nights}$$

| Risk & Value Tier | Trigger Condition | Automated Hotel Action | Business Rationale |
| :--- | :--- | :--- | :--- |
| **Red — High Value at Risk** | $P_{\text{Cancel}} \ge 50\%$ **and** Revenue at Risk $\ge$ €250 | Request partial refundable deposit or credit-card pre-authorization | Protects high-impact suites and multi-night stays where last-minute cancellations hurt most. |
| **Yellow — Moderate Risk** | $P_{\text{Cancel}} \ge 50\%$ **and** Revenue at Risk $<$ €250 | Send automated SMS / email confirmation reminder 7 days prior | Low-cost digital nudge that frees up canceled inventory early without adding checkout friction. |
| **Green — Low Risk** | $P_{\text{Cancel}} < 50\%$ | Instant frictionless confirmation | Protects conversion rates and rewards high-intent direct and short-lead guests. |

---

### Scope Transparency: Working Prototype vs. Proposed Production Rollout

To keep expectations realistic, the table below clearly separates what our Data Science team has **already built and verified in Python** versus what requires **pilot approval and cross-functional engineering**:

| Project Component | Current Status | What Exists / What Is Requested |
| :--- | :--- | :--- |
| **Data Audit & Leakage Removal** | **Built (PoC Ready)** | 119,390 bookings cleaned; 4 post-booking leakage columns dropped; chronological `2015–2016` $\rightarrow$ `2017` split enforced. |
| **Dual ML Engine (Regression + Classification)** | **Built (PoC Ready)** | Automated Scikit-Learn `Pipeline` predicting Fair Nightly Price (`ADR`) and Cancellation Risk (`is_canceled`). |
| **2017 Reality Check & Live Scoring Function** | **Built (PoC Ready)** | Historical backtest proving Safe vs. Risky separation + working `check_booking()` function applying the Red/Yellow/Green policy. |
| **PMS Data Connector & Automated Reminders** | **Proposed (Pilot Scope)** | Engineering middleware to feed live daily PMS bookings into our scoring script and trigger CRM emails/SMS. |
| **Shadow Mode, Front-Desk SOP & Live A/B Test** | **Proposed (Pilot Scope)** | Operational testing on live guests to measure real revenue recovered and verify zero checkout conversion drop. |

---

### Work Breakdown Structure (WBS) & 12-Week Project Schedule

| Phase | Timeline | Status | Key Activities & Milestone Gate |
| :--- | :--- | :--- | :--- |
| **Phase 1: Discovery & Data Audit** | Weeks 1–2 | *Validated in PoC* | Audit historical PMS booking logs; remove zero-guest/zero-night anomalies and strip post-booking leakage fields (`reservation_status`, `reservation_status_date`, `assigned_room_type`, `deposit_type`). **M1:** Clean schema signed off. |
| **Phase 2: Dual-Model Development** | Weeks 3–5 | *Validated in PoC* | Package Regression (`ADR`) and Classification (`is_canceled`) models using strict chronological validation (`2015–2016` $\rightarrow$ `2017`). **M2:** Models meet target MAE and ROC-AUC benchmarks. |
| **Phase 3: Business Policy & Backtest** | Weeks 6–7 | *Validated in PoC* | Confirm Red/Yellow/Green thresholds with Revenue Managers; review 2017 Safe vs. Risky financial backtest. **M3:** Executive sign-off on intervention thresholds. |
| **Phase 4: Shadow Pilot & PMS Integration** | Weeks 8–10 | *Pending Approval* | Connect daily PMS exports to RiskShield Booking AI in "shadow mode" (scoring live bookings without altering guest checkout) and train front-desk leads. **M4:** Zero-downtime shadow run completed. |
| **Phase 5: Controlled Live Rollout (A/B Test)** | Weeks 11–12 | *Pending Approval* | Activate automated Yellow reminders and Red deposit triggers on $50\%$ of eligible bookings (A/B test); publish executive ROI scorecard. **M5:** Production handover & scale decision. |

---

### Project Team Roles & Responsibilities

| Role | Allocation | Core Responsibilities |
| :--- | :--- | :--- |
| **Project Sponsor (VP of Revenue Management)** | 10% | Approves budget, defines acceptable guest-friction thresholds, champions property adoption. |
| **Project Manager** | 50% | Manages 12-week schedule, stakeholder updates, risk register, and cross-property coordination. |
| **Lead Data Scientist** | 100% | Owns the validated PoC, leakage prevention, Regression & Classification modeling, and backtesting. |
| **Data / Integration Engineer** | 75% | Automates daily PMS data extraction, wraps the Python scoring pipeline for live execution, and configures drift alerts. |
| **Hotel Operations Lead (City & Resort)** | 25% | Validates operational workflows, tests deposit/reminder scripts, and gathers front-desk & guest feedback during pilot. |

---

### Risk Register & Mitigation Plan

| Risk ID | Risk Description | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | **Data Leakage in Historical Logs:** Training on fields updated *after* booking (e.g., room reassignments or cancellation timestamps) inflates scores that fail live. | Medium | High | Already mitigated in PoC via a strict 8-feature booking-time schema and chronological validation (`Past` $\rightarrow$ `Future`). |
| **R2** | **Guest Checkout Friction:** Asking for deposits on too many bookings could hurt conversion rates. | Medium | High | Gate Red-tier deposit prompts behind *both* high risk ($\ge 50\%$) and high financial exposure ($\ge$ €250), validated via a controlled A/B test in Weeks 11–12. |
| **R3** | **Seasonal & Market Drift:** Sudden travel shocks, new local festivals, or competitor rate wars may shift cancellation patterns. | Medium | Medium | Schedule automated monthly drift checks comparing predicted vs. actual cancellation rates and trigger quarterly model retraining. |
| **R4** | **PMS Integration Delays:** Legacy booking software may slow real-time API integration. | High | Medium | Launch Phase 4 as a lightweight daily batch report emailed to Revenue Managers each morning—delivering immediate value while live API hooks are finalized. |

---

### Reporting & Governance Tools
* **Weekly Project Tracking:** Jira / Asana milestone board with automated Red-Amber-Green (RAG) status summaries for the Project Sponsor.
* **Model & Data Version Control:** Git repository storing the reproducible Python scoring pipeline and chronological validation scorecards.
* **Operational Reporting:** Bi-weekly Revenue Protection Scorecard tracking actual cancellation rates across Red, Yellow, and Green booking cohorts, deposit conversion rate, and net rooms recovered.

---

## 4. Project Deliverables & Goals

### End Product & Final Objective
The final objective of this initiative is a **production-ready RiskShield Booking AI Engine** deployed across our City and Resort properties, enabling revenue teams to automatically score every incoming booking for Fair Nightly Value (`ADR`) and Cancellation Probability, and trigger proportionate revenue-protection actions.

### Deliverable Timeline

| Deliverable | Description | Target Completion |
| :--- | :--- | :--- |
| **D1. Audited Hospitality Feature Dataset** | Clean, standardized historical booking pipeline stripped of post-booking leakage and invalid anomalies *(PoC Complete)*. | End of Week 2 |
| **D2. Validated Dual-Model ML Pipeline** | Trained and chronologically tested Regression (`ADR`) + Classification (`is_canceled`) Scikit-Learn pipeline *(PoC Complete)*. | End of Week 5 |
| **D3. 2017 Financial Reality-Check Report** | Executive backtest report proving actual cancellation separation and total canceled Euros captured across an unseen full year *(PoC Complete)*. | End of Week 7 |
| **D4. Automated Daily Booking Risk Queue** | Live scoring workflow connecting PMS exports to Red, Yellow, and Green action tiers with clear Euro exposure metrics. | End of Week 10 |
| **D5. Pilot A/B Test Impact Review & Playbook** | Post-rollout financial ROI readout, front-desk SOP guide, and maintenance schedule for multi-property scaling. | End of Week 12 |

### SMART Goals & Critical Success Factors (CSFs)

1. **Model Predictive Reliability (Specific & Measurable):** Achieve a cancellation risk **ROC-AUC $\ge 0.78$** and a nightly room-rate **MAE $\le$ €25/night** on strictly unseen chronological future bookings by Week 5.
2. **Actionable Risk Separation (Measurable & Relevant):** Demonstrate on historical backtesting that bookings classified as **Risky** experience an actual cancellation rate at least **$2\times$ higher** than bookings classified as **Safe** by Week 7.
3. **Revenue Protection Impact (Achievable & Time-Bound):** Reduce unrecovered last-minute cancellations within the Red-tier cohort by **$15\%\text{–}20\%$** during the Weeks 11–12 pilot without reducing overall booking conversion by more than $1.5\%$.
4. **Operational Adoption (Critical Success Factor):** Achieve $\ge 90\%$ daily review and adherence to Red-tier action prompts by City and Resort revenue managers throughout the pilot.

---

## 5. Required Resources & Budget

Because the core Machine Learning engine is built entirely on **open-source Python and Scikit-Learn tooling** (zero proprietary AI license fees), the pilot budget focuses strictly on connecting our working prototype to live hotel operations—keeping total 12-week expenditure to **€42,500**.

### Project Budget & Cost Breakdown

| Cost Category | Estimated Cost (€) | Rationale & Research Justification |
| :--- | :---: | :--- |
| **Internal Team Allocation (Data Science, Engineering, PM)** | €30,000 | Prorated 12-week allocation for Lead Data Scientist, Data Engineer, and Project Manager to harden the ML pipeline, build integrations, and run the A/B pilot. |
| **Cloud Compute, Scheduling & Monitoring Infrastructure** | €3,500 | Managed cloud hosting for automated daily batch scoring, model versioning, and monthly drift monitoring dashboards. |
| **PMS Data Connector & CRM Reminder Automation** | €4,500 | Lightweight integration middleware to read daily booking exports and trigger automated Yellow-tier SMS/email confirmation reminders. |
| **Front-Desk & Revenue Team Training** | €1,500 | Workshops and SOP playbooks ensuring hotel staff communicate deposit requests politely and consistently. |
| **Contingency Reserve ($\sim 7.5\%$)** | €3,000 | Buffer for unexpected PMS schema variations or extended shadow-mode validation. |
| **Total 12-Week Pilot Budget** | **€42,500** | **Break-Even Threshold:** Recovering just **100–120 canceled multi-night stays** across both properties pays back the entire pilot investment within the first quarter. |

### Resource Allocation Plan
* **Weeks 1–5 (Data & ML Hardening — $45\%$ of budget):** Productionizing the PoC feature schema, automated validation checks, and dual-model Scikit-Learn pipeline.
* **Weeks 6–10 (Integration & Shadow Phase — $35\%$ of budget):** PMS connector setup, automated reminder configuration, cloud deployment, and staff training.
* **Weeks 11–12 (Live A/B Pilot & Handover — $20\%$ of budget):** Controlled live testing, executive ROI measurement, and operational documentation.

---

## 6. Conclusion

Unpredictable booking cancellations are not just an operational nuisance—they represent one of the largest controllable revenue leaks in modern hospitality. Relying on gut instinct, blanket non-refundable restrictions, or unprioritized manual phone calls leaves millions of Euros exposed every year while alienating our most reliable guests.

**RiskShield Booking AI** offers a practical, transparent, and already-validated solution built directly on our existing reservation data. By pairing **Dynamic Price Valuation (Regression)** with **Booking-Time Cancellation Risk (Classification)**, the platform pinpoints exactly *which* bookings are likely to cancel and *how many Euros* are truly at stake—empowering our hotels to intervene only where the financial return justifies action.

With a working ML Proof-of-Concept already validated on 119,390 real bookings, a lean pilot investment of **€42,500**, and zero proprietary software lock-in, **RiskShield Booking AI** is positioned to pay for itself rapidly and establish a scalable analytics foundation across the entire Aurelia Hospitality Group portfolio.

**Requested Action:** We respectfully request Executive Committee approval of the **€42,500** pilot budget and authorization to transition our validated Proof-of-Concept into **Phase 4 (Shadow Pilot & PMS Integration)**.
