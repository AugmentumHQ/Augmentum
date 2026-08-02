"""Artifact template registry — professional design templates for AI-generated documents.

When the AI generates an artifact, the system finds the best-matching template
via description similarity and injects its layout instructions as context.
The AI fills in the content; the template defines the design.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArtifactTemplate:
    """A professional template definition for artifact generation."""

    name: str
    description: str           # semantic description for vector matching
    format: str                # pdf, docx, pptx, xlsx, chart
    category: str              # business, academic, creative, technical, data
    layout: dict = field(default_factory=dict)  # format-specific layout instructions
    context_prompt: str = ""   # injected into the AI's system prompt when this template matches


# ---------------------------------------------------------------------------
# PRESENTATION TEMPLATES (PPTX)
# ---------------------------------------------------------------------------

PPTX_CORPORATE_REPORT = ArtifactTemplate(
    name="corporate-report",
    description="Business presentation for quarterly reports, board meetings, investor updates, company overviews, financial reviews",
    format="pptx",
    category="business",
    layout={
        "slide_count_range": [8, 15],
        "suggested_layouts": [
            {"type": "title", "notes": "Company name + presentation title + date + presenter name"},
            {"type": "content", "notes": "Executive summary — 3-4 key takeaways as bullets"},
            {"type": "content", "notes": "Background/context — why this matters"},
            {"type": "two_column", "notes": "Key metrics side by side — numbers on left, context on right"},
            {"type": "content", "notes": "Detailed findings — use sub-bullets for depth"},
            {"type": "content", "notes": "Financial overview or data summary"},
            {"type": "content", "notes": "Challenges and risks — honest assessment"},
            {"type": "content", "notes": "Recommendations and next steps"},
            {"type": "content", "notes": "Timeline or roadmap"},
            {"type": "title", "notes": "Thank you / Q&A slide with contact info"},
        ],
        "design_rules": {
            "max_bullets_per_slide": 5,
            "max_words_per_bullet": 15,
            "use_numbers": True,
            "include_speaker_notes": True,
        },
    },
    context_prompt="""Design a polished corporate presentation. Follow these rules:
- Title slide: Include presenter name, date, and company/team name
- Limit each slide to 4-5 bullet points maximum
- Keep bullets concise (under 15 words each)
- Use specific numbers and metrics where possible
- Include speaker notes for every slide with talking points
- Structure: Executive Summary → Context → Data/Findings → Analysis → Recommendations → Next Steps
- End with a clear call-to-action or Q&A slide
- Two-column layouts work well for before/after comparisons or metric + explanation pairs""",
)

PPTX_PITCH_DECK = ArtifactTemplate(
    name="pitch-deck",
    description="Startup pitch deck, product launch, sales presentation, venture capital pitch, business proposal, partnership proposal",
    format="pptx",
    category="business",
    layout={
        "slide_count_range": [10, 14],
        "suggested_layouts": [
            {"type": "title", "notes": "Company name + tagline + logo placeholder"},
            {"type": "content", "notes": "The Problem — what pain point you solve"},
            {"type": "content", "notes": "The Solution — your product/approach"},
            {"type": "content", "notes": "How It Works — 3 simple steps or key features"},
            {"type": "content", "notes": "Market Opportunity — TAM/SAM/SOM or market size"},
            {"type": "two_column", "notes": "Traction — metrics on left, milestones on right"},
            {"type": "content", "notes": "Business Model — how you make money"},
            {"type": "content", "notes": "Competition — your differentiators"},
            {"type": "content", "notes": "Team — key people and their relevant experience"},
            {"type": "content", "notes": "Financial Projections — 3-year outlook"},
            {"type": "content", "notes": "The Ask — what you need and what you'll do with it"},
            {"type": "title", "notes": "Contact info + next steps"},
        ],
        "design_rules": {
            "max_bullets_per_slide": 4,
            "max_words_per_bullet": 12,
            "use_numbers": True,
            "include_speaker_notes": True,
        },
    },
    context_prompt="""Design a compelling pitch deck. Follow these rules:
- Start with the problem, then solution — hook the audience immediately
- One idea per slide — don't overload
- Use specific numbers: market size in dollars, growth percentages, user counts
- Keep text minimal — slides are visual aids, not documents
- Include speaker notes with the full narrative for each slide
- Traction slide should lead with your strongest metric
- Competition slide: show your unique positioning, not just a feature matrix
- The Ask slide must be specific: amount, use of funds, timeline""",
)

PPTX_EDUCATIONAL = ArtifactTemplate(
    name="educational-lecture",
    description="Educational presentation, lecture slides, teaching materials, workshop, training session, tutorial, course content, seminar",
    format="pptx",
    category="academic",
    layout={
        "slide_count_range": [12, 25],
        "suggested_layouts": [
            {"type": "title", "notes": "Topic title + learning objectives"},
            {"type": "content", "notes": "Agenda/outline of the lecture"},
            {"type": "content", "notes": "Introduction — why this topic matters"},
            {"type": "content", "notes": "Core concept 1 — definition + explanation"},
            {"type": "content", "notes": "Example or case study for concept 1"},
            {"type": "content", "notes": "Core concept 2 — building on concept 1"},
            {"type": "two_column", "notes": "Comparison or contrast of approaches"},
            {"type": "content", "notes": "Core concept 3 — advanced application"},
            {"type": "content", "notes": "Common misconceptions or pitfalls"},
            {"type": "content", "notes": "Practice exercise or discussion question"},
            {"type": "content", "notes": "Summary — key takeaways"},
            {"type": "content", "notes": "Further reading and resources"},
        ],
        "design_rules": {
            "max_bullets_per_slide": 5,
            "max_words_per_bullet": 20,
            "use_numbers": False,
            "include_speaker_notes": True,
        },
    },
    context_prompt="""Design clear educational slides. Follow these rules:
- Start with learning objectives — what will the audience know after this?
- Build concepts incrementally — each slide builds on the previous
- Use examples and analogies to make abstract concepts concrete
- Include a mix of content slides and interactive/discussion slides
- Speaker notes should contain the full explanation the presenter would give
- End with a summary that maps back to the learning objectives
- Include 'Further Reading' or 'Resources' at the end
- For complex topics, use two-column layouts to compare/contrast""",
)

PPTX_TECHNICAL = ArtifactTemplate(
    name="technical-review",
    description="Technical presentation, architecture review, system design, engineering update, code review, technical proposal, RFC, design doc presentation",
    format="pptx",
    category="technical",
    layout={
        "slide_count_range": [8, 16],
        "suggested_layouts": [
            {"type": "title", "notes": "System/feature name + technical context"},
            {"type": "content", "notes": "Problem statement — what technical challenge this addresses"},
            {"type": "content", "notes": "Current architecture or approach"},
            {"type": "content", "notes": "Proposed solution — high-level design"},
            {"type": "content", "notes": "Key design decisions and trade-offs"},
            {"type": "two_column", "notes": "Pros vs cons, or Option A vs Option B"},
            {"type": "content", "notes": "Implementation plan — phases or milestones"},
            {"type": "content", "notes": "Risks and mitigations"},
            {"type": "content", "notes": "Open questions for discussion"},
        ],
        "design_rules": {
            "max_bullets_per_slide": 5,
            "max_words_per_bullet": 18,
            "use_numbers": True,
            "include_speaker_notes": True,
        },
    },
    context_prompt="""Design a clear technical presentation. Follow these rules:
- Lead with the problem before the solution
- Use precise technical terminology but define acronyms on first use
- Include architecture diagrams or system flow descriptions where relevant
- Trade-offs slide is critical — show you considered alternatives
- Implementation plan should have concrete phases with estimates
- Risks section should be honest — technical audiences respect candor
- End with open questions to drive discussion
- Speaker notes should contain additional technical detail""",
)

# ---------------------------------------------------------------------------
# DOCUMENT TEMPLATES (PDF/DOCX)
# ---------------------------------------------------------------------------

PDF_BUSINESS_REPORT = ArtifactTemplate(
    name="business-report",
    description="Business report, quarterly review, annual report, market analysis, competitive analysis, industry report, white paper",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": True,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a professional business report. Structure it as:
1. Cover page with title, author, date, and organization
2. Executive Summary (1 page max — the entire report in miniature)
3. Introduction/Background — context and scope
4. Methodology or Approach (if applicable)
5. Key Findings — use numbered sections with clear headings
6. Analysis — interpret the findings, include data and specific metrics
7. Recommendations — actionable, specific, prioritized
8. Conclusion — brief, ties back to the objective
9. Appendix (if needed)

Use bold text for key terms. Use bullet points for lists of 3+ items. Include specific numbers and data wherever possible. Each section should start with a one-sentence summary.""",
)

PDF_RESEARCH_PAPER = ArtifactTemplate(
    name="research-paper",
    description="Research paper, academic paper, literature review, study results, scientific report, journal article, thesis, dissertation",
    format="pdf",
    category="academic",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": False,
    },
    context_prompt="""Create a well-structured research document. Structure it as:
1. Title page with title, author(s), institution, date
2. Abstract (150-300 words — objective, methods, results, conclusions)
3. Introduction — background, research question, significance
4. Literature Review or Related Work
5. Methodology — detailed enough to reproduce
6. Results — present data and findings objectively
7. Discussion — interpret results, compare with existing work, limitations
8. Conclusion — summarize findings, implications, future work
9. References (list sources mentioned)

Use formal academic tone. Be precise with terminology. Present data before interpretation. Acknowledge limitations explicitly.""",
)

PDF_PROJECT_PROPOSAL = ArtifactTemplate(
    name="project-proposal",
    description="Project proposal, project plan, project charter, implementation plan, strategy document, initiative proposal, budget proposal",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": True,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a compelling project proposal. Structure it as:
1. Cover page with project name, proposed by, date, version
2. Executive Summary — the proposal in one page
3. Problem Statement — what problem this project solves
4. Proposed Solution — high-level approach
5. Scope — what's included and what's explicitly NOT included
6. Timeline — phases with milestones and dates
7. Budget — itemized costs with totals
8. Team and Resources — who's involved and their roles
9. Risk Assessment — identified risks with mitigation strategies
10. Success Metrics — how you'll measure success
11. Next Steps — immediate actions needed

Be specific with timelines, costs, and metrics. Use tables for budget and timeline. Include both best-case and realistic estimates.""",
)

PDF_TECHNICAL_DOC = ArtifactTemplate(
    name="technical-documentation",
    description="Technical documentation, API documentation, system documentation, architecture document, design document, specification, technical manual, user guide",
    format="pdf",
    category="technical",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create clear technical documentation. Structure it as:
1. Title page with document name, version, date, author
2. Table of Contents
3. Overview — what this document covers and who it's for
4. Prerequisites — what the reader needs to know or have
5. Architecture/Design — high-level system description
6. Detailed Sections — organized by component or workflow
7. Configuration — settings, parameters, environment variables
8. Troubleshooting — common issues and solutions
9. Glossary — define technical terms
10. Changelog — version history

Use code blocks for commands and configuration. Use numbered steps for procedures. Include examples for every API endpoint or configuration option. Mark required vs optional parameters.""",
)

# ---------------------------------------------------------------------------
# SPREADSHEET TEMPLATES (XLSX)
# ---------------------------------------------------------------------------

XLSX_FINANCIAL = ArtifactTemplate(
    name="financial-spreadsheet",
    description="Financial spreadsheet, budget tracker, P&L statement, revenue forecast, expense report, financial model, cash flow, balance sheet",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": True,
        "use_percentage_format": True,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create a well-structured financial spreadsheet. Follow these rules:
- Use currency formatting ($#,##0.00) for all monetary values
- Use percentage formatting (0.0%) for rates and changes
- Include a totals/summary row at the bottom
- Add a 'Variance' or 'Change' column comparing to budget/previous period
- Use consistent column order: Label | Current | Previous | Change | % Change
- Headers should be clear and unambiguous
- Include formulas for calculated fields (totals, percentages, variances)
- Round to appropriate precision (dollars to cents, percentages to one decimal)""",
)

XLSX_PROJECT_TRACKER = ArtifactTemplate(
    name="project-tracker",
    description="Project tracker, task list, sprint board, milestone tracker, project status, Gantt chart data, resource allocation, workload tracker",
    format="xlsx",
    category="business",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": True,
        "include_summary_row": False,
        "summary_type": "count",
    },
    context_prompt="""Create a clear project tracking spreadsheet. Follow these rules:
- Columns: Task | Owner | Status | Priority | Start Date | Due Date | % Complete | Notes
- Status values should be consistent: Not Started, In Progress, Complete, Blocked
- Priority values: High, Medium, Low
- Dates in YYYY-MM-DD format
- % Complete as decimal (0.0 to 1.0) for percentage formatting
- Sort by priority (High first) then by due date
- Use a summary row counting tasks by status if multiple sheets""",
)

XLSX_COMPARISON = ArtifactTemplate(
    name="comparison-matrix",
    description="Comparison matrix, feature comparison, vendor evaluation, product comparison, scoring rubric, decision matrix, alternatives analysis",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": False,
        "include_summary_row": True,
        "summary_type": "average",
    },
    context_prompt="""Create a clear comparison spreadsheet. Follow these rules:
- First column: Criteria/Feature names
- Subsequent columns: one per option being compared
- Use a consistent scoring system (1-5, or Yes/No, or specific values)
- Include a 'Weight' column if criteria have different importance
- Add a weighted score total at the bottom
- Group related criteria with blank rows or subheaders
- Include a 'Notes' column for qualitative observations
- Bold the winner/best value in each row""",
)

# ---------------------------------------------------------------------------
# CHART TEMPLATES
# ---------------------------------------------------------------------------

CHART_TREND = ArtifactTemplate(
    name="trend-analysis",
    description="Trend chart, time series, growth chart, historical data, progress over time, performance trend, metric tracking",
    format="chart",
    category="data",
    layout={
        "preferred_type": "line",
        "show_values": False,
        "include_trend_line": True,
    },
    context_prompt="""Create a clear trend visualization. Follow these rules:
- Use a line chart for time-series data
- X-axis should be chronological (dates, months, quarters, years)
- Y-axis should start at a meaningful baseline (not always zero — choose based on data range)
- Label axes clearly with units
- If comparing multiple series, use distinct colors and include a legend
- Keep to 3 or fewer series for readability
- Use descriptive series names, not 'Series 1'""",
)

CHART_COMPARISON = ArtifactTemplate(
    name="comparison-chart",
    description="Comparison chart, bar chart, category comparison, A/B comparison, benchmark, ranking, performance comparison",
    format="chart",
    category="data",
    layout={
        "preferred_type": "bar",
        "show_values": True,
        "include_trend_line": False,
    },
    context_prompt="""Create a clear comparison visualization. Follow these rules:
- Use a bar chart for comparing categories
- If values are very different in magnitude, consider horizontal bars
- Show data values on or near each bar
- Sort bars by value (largest first) unless there's a natural order
- Use a single color for single-series, multiple colors for multi-series
- Label the Y-axis with units
- Keep category labels short (abbreviate if needed)""",
)

CHART_DISTRIBUTION = ArtifactTemplate(
    name="distribution-chart",
    description="Distribution chart, pie chart, market share, composition, breakdown, allocation, proportion, percentage split",
    format="chart",
    category="data",
    layout={
        "preferred_type": "pie",
        "show_values": True,
        "include_trend_line": False,
    },
    context_prompt="""Create a clear distribution visualization. Follow these rules:
- Use a pie chart only for showing parts of a whole (must sum to 100%)
- Limit to 5-7 slices maximum — group small values into 'Other'
- Show percentage labels on each slice
- Order slices from largest to smallest (clockwise from top)
- Use distinct, contrasting colors
- If there are more than 7 categories, use a horizontal bar chart instead""",
)


# ---------------------------------------------------------------------------
# CAREER TEMPLATES (PDF/DOCX)
# ---------------------------------------------------------------------------

PDF_RESUME = ArtifactTemplate(
    name="resume",
    description="Resume, CV, curriculum vitae, professional profile, job application, career summary",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": False,
    },
    context_prompt="""Create a professional resume. Structure it as:
1. Contact Information — full name, phone, email, LinkedIn, city/state (no full address)
2. Professional Summary — 3-4 lines highlighting years of experience, key expertise, and career focus
3. Experience — reverse chronological order. For each role: Company | Title | Dates. 3-5 bullet points per role using strong action verbs (Led, Designed, Increased, Reduced). Quantify achievements with metrics (percentages, dollar amounts, team sizes, time saved)
4. Education — degree, institution, graduation year. Include GPA only if recent grad and above 3.5
5. Skills — grouped by category (Technical, Languages, Tools). List only skills relevant to target role

Keep to 1-2 pages maximum. Use consistent date formatting (Mon YYYY). Every bullet should show impact, not just responsibility. Prioritize recent and relevant experience.""",
)

PDF_COVER_LETTER = ArtifactTemplate(
    name="cover-letter",
    description="Cover letter, application letter, letter of interest, job application letter",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "letter",
        "include_callout_boxes": False,
    },
    context_prompt="""Create a professional cover letter. Structure it as:
1. Header — your contact info and date
2. Addressee — hiring manager by name if possible, company name, address
3. Opening paragraph — hook the reader immediately. Name the specific position. State why you are excited about this role and company (show you researched them)
4. Body paragraph 1 — map your most relevant experience to the job requirements. Use a specific example with measurable results
5. Body paragraph 2 — highlight a second key qualification or transferable skill. Show cultural fit
6. Closing paragraph — express enthusiasm, restate your value proposition, include a clear call to action (interview request)
7. Sign-off — Professional closing (Sincerely), full name

Keep under 1 page. Tone: confident but not arrogant, enthusiastic but professional. Never repeat the resume verbatim — add context and narrative the resume cannot convey.""",
)

# ---------------------------------------------------------------------------
# HR TEMPLATES (PDF/DOCX)
# ---------------------------------------------------------------------------

PDF_JOB_DESCRIPTION = ArtifactTemplate(
    name="job-description",
    description="Job description, job posting, position description, role description, hiring document",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a professional job description. Structure it as:
1. Job Title — clear, standard title (avoid internal jargon)
2. Department and Reports To — where this role sits in the organization
3. Job Summary — 2-3 sentences describing the role and its impact
4. Key Responsibilities — 8-12 bullet points, ordered by importance. Start each with an action verb. Be specific about what the person will DO, not just what the team does
5. Required Qualifications — minimum education, years of experience, must-have skills and certifications
6. Preferred Qualifications — nice-to-have skills, additional experience, bonus certifications
7. Compensation Range — salary band, bonus structure if applicable
8. Benefits — top 5-8 benefits (health, PTO, retirement, remote policy, etc.)
9. Equal Employment Opportunity statement

Use inclusive language. Avoid unnecessary requirements that reduce the candidate pool. Be honest about the role — set accurate expectations.""",
)

PDF_OFFER_LETTER = ArtifactTemplate(
    name="offer-letter",
    description="Offer letter, employment offer, job offer letter, hiring letter",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "letter",
        "include_callout_boxes": False,
    },
    context_prompt="""Create a professional offer letter. Structure it as:
1. Company header/letterhead — company name, address, logo placeholder
2. Date and candidate address
3. Opening — congratulations and expression of enthusiasm for the candidate joining
4. Position Details — title, department, reporting manager, start date, work location (on-site/hybrid/remote)
5. Compensation — base salary (annual), pay frequency, bonus structure and target percentage, equity if applicable
6. Benefits Summary — health insurance, retirement/401k match, PTO days, other key perks
7. At-Will Employment statement (or contract terms if applicable)
8. Contingencies — background check, drug screening, proof of work authorization, reference verification
9. Acceptance Deadline — specific date by which the candidate must respond
10. Signature Blocks — company representative (name, title, signature line) and candidate (signature, printed name, date)

Tone: warm and welcoming but legally precise. Include all material terms. State clearly that this letter is not a contract unless specified otherwise.""",
)

PDF_PERFORMANCE_REVIEW = ArtifactTemplate(
    name="performance-review",
    description="Performance review, employee evaluation, annual review, performance appraisal, self-assessment",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a professional performance review document. Structure it as:
1. Employee Information — name, title, department, manager, review period dates
2. Goals from Previous Period — list each goal with: description, target metric, actual result, rating (1-5 scale: 1=Did Not Meet, 2=Partially Met, 3=Met, 4=Exceeded, 5=Far Exceeded)
3. Core Competency Ratings — rate each on 1-5 scale: Communication, Teamwork, Problem Solving, Initiative, Technical Skills, Leadership (if applicable). Include a brief justification for each rating
4. Key Strengths — 3-4 specific strengths with examples of how they were demonstrated
5. Areas for Improvement — 2-3 specific areas with constructive, actionable suggestions
6. Development Plan — specific training, mentoring, or stretch assignments for the next period
7. Goals for Next Period — 3-5 SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)
8. Employee Comments Section — space for the employee to respond
9. Signatures — employee, manager, and date lines

Be specific with examples. Avoid vague praise or criticism. Every rating should be supported by observable behavior or results.""",
)

PDF_EMPLOYEE_HANDBOOK = ArtifactTemplate(
    name="employee-handbook",
    description="Employee handbook, company policies, workplace policies, HR manual, code of conduct",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a comprehensive employee handbook. Structure it as:
1. Welcome Message — from CEO/founder, company mission and values
2. Employment Classification — full-time, part-time, contractor definitions, at-will statement
3. Code of Conduct — professional behavior expectations, dress code, social media policy
4. Work Schedule and Attendance — hours, flexible work policy, remote work guidelines, absence reporting
5. Compensation and Benefits — pay schedule, overtime, health insurance, retirement plans, other perks
6. Leave Policies — PTO accrual, sick leave, parental leave, bereavement, jury duty, holidays list
7. Workplace Safety — emergency procedures, reporting hazards, workers compensation
8. IT and Security — acceptable use policy, data protection, password requirements, equipment policy
9. Anti-Discrimination and Harassment — zero-tolerance policy, protected classes, reporting procedures, investigation process
10. Grievance and Complaint Procedure — chain of reporting, HR contact, whistleblower protections
11. Acknowledgment Page — employee signature confirming receipt and understanding

Use clear, plain language. Each section should state the policy, then the procedure. Include effective date and revision number.""",
)

PDF_TRAINING_MANUAL = ArtifactTemplate(
    name="training-manual",
    description="Training manual, training guide, onboarding guide, user guide, how-to guide, instruction manual",
    format="pdf",
    category="technical",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a clear, structured training manual. Structure it as:
1. Title page — manual title, version, date, intended audience
2. Purpose and Scope — what this manual covers and who it is for
3. Prerequisites — what the reader should know or have before starting
4. Learning Objectives — what the reader will be able to do after completing each module
5. Content Modules — for each module include:
   a. Concept explanation with real-world context
   b. Step-by-step procedures (numbered, specific, verifiable)
   c. Worked examples with screenshots or diagrams described
   d. Practice exercises with expected outcomes
   e. Common mistakes and how to avoid them
6. Assessment Questions — knowledge checks per module (multiple choice or short answer with answer key)
7. Quick Reference Card — one-page summary of key procedures and shortcuts
8. Glossary — define all technical terms and acronyms
9. Troubleshooting — common problems and their solutions in a problem/cause/fix format

Write for the beginner. Never assume knowledge not listed in prerequisites. Use consistent formatting for all procedures.""",
)

# ---------------------------------------------------------------------------
# LEGAL TEMPLATES (PDF/DOCX)
# ---------------------------------------------------------------------------

PDF_NDA = ArtifactTemplate(
    name="nda",
    description="NDA, non-disclosure agreement, confidentiality agreement, secrecy agreement",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": False,
    },
    context_prompt="""Create a professional non-disclosure agreement. Structure it as:
1. Title — NON-DISCLOSURE AGREEMENT
2. Preamble — effective date, parties (Disclosing Party and Receiving Party) with full legal names and addresses
3. Definition of Confidential Information — broad but bounded definition covering documents, data, trade secrets, business plans, technical information. Specify forms: written, oral, electronic, visual
4. Obligations of Receiving Party — hold in confidence, limit access to need-to-know, use same care as own confidential info (no less than reasonable care), no reverse engineering
5. Exclusions from Confidential Information — publicly available info, already known prior to disclosure, independently developed, received from third party without restriction
6. Term and Duration — period of agreement and survival period for obligations (typically 2-5 years after termination)
7. Return of Materials — obligation to return or destroy all confidential materials upon request or termination
8. Remedies — acknowledge that breach may cause irreparable harm, injunctive relief available
9. Governing Law — jurisdiction for disputes
10. Entire Agreement — this supersedes prior agreements on the subject
11. Signature Blocks — both parties with name, title, date, signature line

Use formal legal language. Include section numbering for easy reference. Note: this is a template — recommend legal review before execution.""",
)

PDF_SERVICE_AGREEMENT = ArtifactTemplate(
    name="service-agreement",
    description="Service agreement, contract, consulting agreement, freelance contract, statement of work, master services agreement",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": False,
    },
    context_prompt="""Create a professional service agreement. Structure it as:
1. Title — SERVICE AGREEMENT or CONSULTING AGREEMENT
2. Parties — full legal names, addresses, and roles (Client and Service Provider)
3. Scope of Services — detailed description of services to be performed, deliverables with acceptance criteria
4. Timeline — project phases, milestones with specific dates, final delivery date
5. Payment Terms — total amount or rate (hourly/daily/fixed), payment schedule (e.g., 50% upfront, 50% on completion), invoicing process, payment method, late payment penalties
6. Intellectual Property — who owns the work product (typically Client upon full payment), pre-existing IP retained by Provider, license grants
7. Confidentiality — mutual confidentiality obligations
8. Warranties — Provider warrants professional standard of care, original work, no infringement
9. Limitation of Liability — cap on damages (typically contract value), exclusion of consequential damages
10. Termination — termination for convenience (notice period), termination for cause, payment for work completed
11. Dispute Resolution — negotiation first, then mediation or arbitration, governing law
12. Force Majeure — unforeseeable events excusing performance
13. General Provisions — entire agreement, amendments in writing, severability, assignment
14. Signature Blocks — both parties with name, title, date

Note: this is a template — recommend legal review before execution.""",
)

# ---------------------------------------------------------------------------
# OPERATIONS TEMPLATES (PDF/DOCX)
# ---------------------------------------------------------------------------

PDF_MEETING_MINUTES = ArtifactTemplate(
    name="meeting-minutes",
    description="Meeting minutes, meeting notes, meeting summary, meeting record, action items",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create clear, actionable meeting minutes. Structure it as:
1. Meeting Header — title, date, time (start and end), location (or virtual platform)
2. Attendees — list present members and note any absent members who were expected
3. Agenda Items — for each agenda item include:
   a. Topic heading
   b. Brief summary of discussion (key points only, not a transcript)
   c. Decisions made (clearly marked)
   d. Action items arising (clearly marked with: What | Who | Due Date)
4. Action Items Summary — consolidated table of ALL action items: Action | Owner | Due Date | Status
5. Next Meeting — date, time, location, preliminary agenda items

Keep concise. Focus on decisions and actions, not the full discussion. Use past tense. Attribute decisions to the group, not individuals, unless a specific person made a commitment.""",
)

PDF_STATUS_REPORT = ArtifactTemplate(
    name="status-report",
    description="Status report, weekly update, progress report, project update, standup notes, weekly summary",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a clear project status report. Structure it as:
1. Header — project name, reporting period (dates), report date, author
2. Overall Status — single indicator: On Track (green), At Risk (yellow), or Blocked (red) with one-sentence justification
3. Accomplishments This Period — 3-5 bullet points of completed work, each with measurable outcome where possible
4. Planned for Next Period — 3-5 bullet points of upcoming work with target completion dates
5. Risks and Blockers — each risk should include: description, impact, mitigation plan, owner. Blockers should include: what is blocked, what is needed to unblock, who can help
6. Key Metrics/KPIs — 3-5 project health metrics (budget spent vs planned, tasks completed vs planned, velocity, defect count, etc.)
7. Help Needed — specific requests for decisions, resources, or escalations

Be factual, not optimistic. Flag problems early. Every item should be specific enough that someone reading it knows exactly what happened or what needs to happen.""",
)

PDF_SOP = ArtifactTemplate(
    name="sop",
    description="SOP, standard operating procedure, process document, workflow document, procedure manual, work instruction",
    format="pdf",
    category="technical",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a clear standard operating procedure. Structure it as:
1. Title Page — SOP title, document number, version, effective date, department
2. Purpose — why this procedure exists (one paragraph)
3. Scope — what this SOP covers and does not cover, who must follow it
4. Responsibilities — who performs each role in this procedure
5. Prerequisites and Safety — required materials, tools, PPE, warnings, precautions
6. Procedure — step-by-step instructions, numbered sequentially:
   - Each step should be a single action
   - Include expected outcome or verification for critical steps
   - Note decision points (if X, go to step Y; if Z, go to step W)
   - Mark critical steps with a warning indicator
7. Expected Outcomes — what success looks like at the end of the procedure
8. Troubleshooting — common problems in a table: Symptom | Possible Cause | Corrective Action
9. Related Documents — references to other SOPs, forms, or standards
10. Revision History — table: Version | Date | Author | Changes
11. Approval — name, title, signature line, date for approver

Write so that someone unfamiliar with the process can follow it correctly on the first attempt.""",
)

# ---------------------------------------------------------------------------
# MARKETING TEMPLATES (PDF/DOCX/XLSX)
# ---------------------------------------------------------------------------

PDF_CASE_STUDY = ArtifactTemplate(
    name="case-study",
    description="Case study, success story, customer story, client testimonial, use case",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": True,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a compelling case study. Structure it as a narrative:
1. Title — attention-grabbing, results-focused (e.g., "How [Client] Reduced Costs by 40%")
2. Client Overview — company name, industry, size, relevant context (1-2 sentences)
3. The Challenge — specific problem the client faced. Include metrics that quantify the pain (cost, time lost, error rates). Make the reader feel the problem
4. The Solution — what was implemented, how it works, why this approach was chosen. Be specific about the product/service delivered
5. The Results — specific, measurable outcomes. Use concrete numbers: percentage improvements, dollar amounts saved, time reduced, revenue gained. Include at least 3 quantified results
6. Client Quote — a testimonial that captures the transformation in the client's own words
7. Key Takeaways — 2-3 lessons or insights that other organizations can apply

Tell a story: Situation leads to Action leads to Result. Use callout boxes for key statistics. The results section is the most important — lead with the biggest number.""",
)

PDF_ONE_PAGER = ArtifactTemplate(
    name="one-pager",
    description="One-pager, business summary, company overview, product brief, executive brief, fact sheet",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a concise, high-impact one-pager. EVERYTHING must fit on a single page. Structure it as:
1. Company/Product Name and Tagline — bold, prominent, memorable
2. The Problem — 1-2 sentences defining the pain point you solve
3. The Solution — 1-2 sentences explaining your approach
4. Key Features or Capabilities — 3-5 items, each with a short label and one-sentence description
5. Target Market — who this is for (be specific)
6. Differentiation — what makes this unique vs alternatives (2-3 points)
7. Traction/Proof Points — key metrics, notable customers, awards, or growth stats
8. Team Highlights — founder/leadership credentials in 1 line each (optional, space permitting)
9. Contact Information — website, email, phone

Visual hierarchy is critical. A reader should grasp the core value proposition in 30 seconds by scanning headings and bold text. Use short sentences and bullet points. No paragraphs longer than 2 sentences.""",
)

PDF_NEWSLETTER = ArtifactTemplate(
    name="newsletter",
    description="Newsletter, email newsletter, company update, monthly update, digest, bulletin",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create an engaging newsletter. Structure it as:
1. Header — publication name/branding, issue number, date
2. Greeting — brief, warm opening addressing the audience
3. Lead Story — feature article (200-300 words) with a compelling headline. This is the main content piece
4. Secondary Stories — 2-3 shorter items (75-100 words each) covering news, updates, or tips
5. Upcoming Events — dates, descriptions, and registration/RSVP information
6. Call-to-Action — one clear action you want the reader to take (sign up, attend, try, share)
7. Footer — unsubscribe note, social media links, contact information, company address

Tone: conversational and approachable, not corporate-stiff. Use scannable formatting — subheadings, bold key phrases, short paragraphs. Each section should be self-contained so skimmers get value. Lead with the most interesting content.""",
)

XLSX_CONTENT_CALENDAR = ArtifactTemplate(
    name="content-calendar",
    description="Content calendar, social media calendar, editorial calendar, publishing schedule, content plan",
    format="xlsx",
    category="business",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": False,
        "include_summary_row": False,
        "summary_type": "count",
    },
    context_prompt="""Create a content calendar spreadsheet. Use these columns:
- Date — publish date in YYYY-MM-DD format
- Platform — target platform (Instagram, Twitter/X, LinkedIn, Blog, YouTube, Email, etc.)
- Content Type — post, story, reel, article, video, infographic, poll, etc.
- Topic — subject or theme of the content piece
- Copy/Caption — draft text or description of the content
- Visual Asset Needed — description of image, video, or graphic required (or "None")
- Status — Draft, In Review, Approved, Scheduled, Published
- Publish Time — target time in HH:MM format with timezone
- Notes — additional context, hashtags, links, or collaboration notes

One row per piece of content. Sort by date. Use consistent status values. Plan at least 2-4 weeks ahead. Mix content types and platforms for variety.""",
)

# ---------------------------------------------------------------------------
# SALES TEMPLATES (PDF/DOCX)
# ---------------------------------------------------------------------------

PDF_SALES_PROPOSAL = ArtifactTemplate(
    name="sales-proposal",
    description="Sales proposal, client proposal, service proposal, pricing proposal, bid, RFP response",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": True,
        "has_toc": True,
        "has_executive_summary": True,
        "section_style": "numbered",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a persuasive sales proposal. Structure it as:
1. Cover Page — client name prominently displayed, your company name, proposal title, date, confidentiality notice
2. Executive Summary — 1 page max. Restate the client's need, your proposed solution, key benefits, and investment summary. The client should be able to make a decision from this page alone
3. Understanding of Needs — demonstrate you listened. Restate the client's challenges, goals, and requirements in their language. Reference specific conversations or RFP items
4. Proposed Solution — what you will deliver, how it addresses each stated need, key features and benefits. Focus on outcomes, not features
5. Implementation Approach — methodology, phases, timeline with milestones, client responsibilities
6. Timeline — visual or table format: Phase | Activities | Duration | Deliverables
7. Pricing — detailed table: Line Item | Description | Quantity | Unit Price | Total. Include subtotal, any discounts, taxes, grand total. State payment terms
8. Terms and Conditions — validity period, payment schedule, warranties, limitations
9. Team — key team members with brief bios and relevant experience
10. Case Studies/References — 2-3 relevant examples of similar work with results
11. Next Steps — clear actions to move forward, with proposed dates

Make the client feel understood. Lead with their goals, not your capabilities.""",
)

# ---------------------------------------------------------------------------
# ACADEMIC TEMPLATES (PDF/DOCX/XLSX)
# ---------------------------------------------------------------------------

PDF_LESSON_PLAN = ArtifactTemplate(
    name="lesson-plan",
    description="Lesson plan, teaching plan, class plan, instructional plan, unit plan",
    format="pdf",
    category="academic",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a structured lesson plan. Structure it as:
1. Header — subject, grade level/audience, date, time allocation (total minutes)
2. Learning Objectives — 2-4 measurable objectives using Bloom's taxonomy verbs (identify, analyze, create, evaluate). Students should be able to [verb] [what] [to what standard]
3. Standards Alignment — relevant curriculum standards or competency frameworks
4. Materials Needed — list all supplies, handouts, technology, and preparation required
5. Lesson Sequence:
   a. Warm-Up/Hook (5 min) — engage students, activate prior knowledge, pose a question
   b. Direct Instruction (15 min) — present new content, model skills, use examples
   c. Guided Practice (15 min) — students practice with teacher support, check for understanding
   d. Independent Practice (10 min) — students apply learning independently
   e. Closure (5 min) — summarize key points, exit ticket or reflection question
6. Assessment — how you will measure whether objectives were met (formative and/or summative)
7. Differentiation — modifications for advanced learners (extension activities) and struggling learners (scaffolding, accommodations)

Time allocations should sum to the total class period. Include transition instructions between activities.""",
)

XLSX_RUBRIC = ArtifactTemplate(
    name="rubric",
    description="Rubric, grading rubric, scoring rubric, assessment rubric, evaluation criteria, grading matrix",
    format="xlsx",
    category="academic",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": False,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create a detailed grading rubric spreadsheet. Use these columns:
- Criteria — the skill or dimension being evaluated (one per row)
- Excellent (4) — specific description of what excellent performance looks like for this criterion
- Good (3) — specific description of good/proficient performance
- Adequate (2) — specific description of developing/basic performance
- Needs Improvement (1) — specific description of beginning/insufficient performance
- Weight — relative importance of this criterion (e.g., 1x, 2x, or percentage)

Each cell should contain specific, observable descriptors — not vague words like "good" or "adequate." Describe what the student DID, not what they ARE. Include a Total Points row at the bottom showing maximum possible score. Use quantitative thresholds where possible (e.g., "cites 3+ sources" vs "cites 1 source").""",
)

PDF_SYLLABUS = ArtifactTemplate(
    name="syllabus",
    description="Syllabus, course outline, course syllabus, class schedule, course description",
    format="pdf",
    category="academic",
    layout={
        "has_cover_page": False,
        "has_toc": True,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a comprehensive course syllabus. Structure it as:
1. Course Information — course title, number, section, term/semester, meeting days/times, location
2. Instructor Information — name, email, office location, office hours (days/times), preferred contact method
3. Course Description — 3-5 sentence overview of the course content and purpose
4. Learning Outcomes — 4-6 specific, measurable outcomes. Upon completion, students will be able to...
5. Required Texts and Materials — books (author, title, edition, ISBN), software, supplies
6. Grading Breakdown — table format: Component | Percentage | Description (e.g., Participation 10%, Midterm Exam 20%, Final Project 30%, Homework 25%, Quizzes 15%). Must total 100%
7. Course Policies:
   a. Attendance — expectations and impact on grade
   b. Late Work — penalty structure (e.g., 10% per day, max 3 days)
   c. Academic Integrity — plagiarism definition, consequences, honor code reference
   d. Accommodations — disability services contact information
8. Weekly Schedule — table: Week | Dates | Topic | Readings/Assignments Due
9. Important Dates — add/drop deadline, midterm date, final exam date, project due dates

Be thorough but readable. Students refer to this document all semester.""",
)

# ---------------------------------------------------------------------------
# FINANCIAL TEMPLATES (XLSX)
# ---------------------------------------------------------------------------

XLSX_INVOICE = ArtifactTemplate(
    name="invoice",
    description="Invoice, bill, receipt, payment request, billing statement",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": True,
        "use_percentage_format": True,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create a professional invoice spreadsheet. Structure it as:
- Header section: Invoice Number, Invoice Date, Due Date, Payment Terms (e.g., Net 30)
- Bill From: company name, address, phone, email, tax ID (if applicable)
- Bill To: client name, company, address, contact email
- Line Items table with columns: Item # | Description | Quantity | Unit Price | Amount
- Each Amount = Quantity x Unit Price
- Below the line items: Subtotal, Tax Rate (%), Tax Amount, Total Due
- Payment Information: accepted methods, bank details or PayPal address, check payable to
- Notes/Terms: late payment policy, thank-you message

Use currency formatting ($#,##0.00) for all monetary values. Include formulas for Amount, Subtotal, Tax, and Total. Number all line items sequentially.""",
)

XLSX_BUDGET_REPORT = ArtifactTemplate(
    name="budget-report",
    description="Budget report, budget vs actual, variance report, financial summary, spending report",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": True,
        "use_percentage_format": True,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create a budget vs. actual report spreadsheet. Use these columns:
- Category — budget line item name
- Budget — planned amount for the period
- Actual — actual spend for the period
- Variance — Actual minus Budget (negative means under budget)
- % Variance — Variance divided by Budget, as percentage

Group rows by department or expense type (e.g., Personnel, Operations, Marketing, Technology, Facilities). Include subtotals for each group and a grand total row at the bottom. Use currency formatting for dollar amounts and percentage formatting for variance %. Negative variance (over budget) should be clearly distinguishable. Include YTD (Year-to-Date) columns if tracking a multi-period budget: YTD Budget | YTD Actual | YTD Variance | YTD % Variance.""",
)

XLSX_EXPENSE_REPORT = ArtifactTemplate(
    name="expense-report",
    description="Expense report, reimbursement form, travel expenses, receipt tracker",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": True,
        "use_percentage_format": False,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create an expense report spreadsheet. Structure it as:
- Header section: Employee Name, Department, Manager, Reporting Period (start-end dates)
- Line items with columns: Date | Description | Category | Amount | Receipt Attached (Y/N)
- Categories: Travel, Meals, Lodging, Transportation, Supplies, Communication, Other
- Subtotals by category at the bottom
- Grand Total of all expenses
- Approval section: Employee Signature/Date, Manager Approval/Date, Finance Approval/Date

Use currency formatting ($#,##0.00) for all amounts. Sort by date within the report. Include a separate summary section that totals expenses by category. Every expense should have a date and description specific enough for audit purposes.""",
)

# ---------------------------------------------------------------------------
# DATA/ANALYTICS TEMPLATES (XLSX)
# ---------------------------------------------------------------------------

XLSX_KPI_DASHBOARD = ArtifactTemplate(
    name="kpi-dashboard",
    description="KPI dashboard, metrics dashboard, scorecard, performance dashboard, analytics report",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": True,
        "include_summary_row": False,
        "summary_type": "count",
    },
    context_prompt="""Create a KPI dashboard spreadsheet. Use these columns:
- KPI Name — clear, specific metric name
- Target — the goal value for this period
- Actual — the achieved value for this period
- % of Target — Actual divided by Target, as percentage
- Status — On Track (>=90% of target), At Risk (70-89%), Behind (<70%)
- Trend — Up, Down, or Flat compared to previous period
- Previous Period — last period's actual value for comparison
- Period — the time frame (e.g., Q1 2026, March 2026)

Group KPIs by department or strategic objective (e.g., Revenue, Customer, Operations, People). Use percentage formatting for rate-based KPIs and appropriate number formatting for counts and currency. Status column should use clear text labels. Include a summary at the top: total KPIs on track, at risk, and behind.""",
)

XLSX_SURVEY_RESULTS = ArtifactTemplate(
    name="survey-results",
    description="Survey results, poll results, feedback summary, questionnaire analysis, customer feedback",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": True,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create a survey results spreadsheet. Structure it as:
- Summary section at top: Total Respondents, Response Rate, Survey Period
- For each question, use columns: Question | Response Option | Count | Percentage
- Group rows by survey section or theme
- For Likert scale questions (Strongly Agree to Strongly Disagree): include a Mean Score column (1-5 scale)
- For multiple-choice questions: show all options with counts and percentages
- For open-ended questions: list Top Themes with frequency counts
- Include a summary statistics section: overall satisfaction score, Net Promoter Score if applicable, key findings (3-5 bullets as text)

Percentages should sum to 100% within each question. Sort response options by frequency (highest first) unless there is a natural order (like Likert scales). Use percentage formatting for all rate columns.""",
)

XLSX_INVENTORY_TRACKER = ArtifactTemplate(
    name="inventory-tracker",
    description="Inventory tracker, stock tracker, warehouse inventory, product catalog, stock levels",
    format="xlsx",
    category="data",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": True,
        "use_percentage_format": False,
        "include_summary_row": True,
        "summary_type": "sum",
    },
    context_prompt="""Create an inventory tracking spreadsheet. Use these columns:
- Item ID — unique identifier (e.g., SKU or part number)
- Name — product or item name
- Category — product category or department
- Qty On Hand — current stock quantity
- Reorder Point — minimum quantity that triggers a reorder
- Reorder Qty — how many to order when reorder point is reached
- Unit Cost — cost per unit in currency format
- Total Value — Qty On Hand multiplied by Unit Cost
- Supplier — supplier or vendor name
- Last Ordered — date of most recent order (YYYY-MM-DD)
- Status — In Stock (above reorder point), Low Stock (at or below reorder point), Out of Stock (zero quantity)

Sort by category, then by name. Include a totals row for Total Value column. Status should be derived from Qty On Hand vs Reorder Point. Flag items that need immediate reordering.""",
)

# ---------------------------------------------------------------------------
# STRATEGIC TEMPLATES (PDF/XLSX)
# ---------------------------------------------------------------------------

PDF_SWOT_ANALYSIS = ArtifactTemplate(
    name="swot-analysis",
    description="SWOT analysis, strategic analysis, competitive analysis framework, strengths weaknesses opportunities threats",
    format="pdf",
    category="business",
    layout={
        "has_cover_page": False,
        "has_toc": False,
        "has_executive_summary": False,
        "section_style": "headed",
        "include_callout_boxes": True,
    },
    context_prompt="""Create a SWOT analysis document. Structure it as a 2x2 grid:
1. Title — SWOT Analysis for [Subject], date
2. Strengths (Internal, Positive) — 4-6 bullet points identifying internal advantages: competitive edges, strong resources, unique capabilities, brand reputation, talent. Be specific and evidence-based
3. Weaknesses (Internal, Negative) — 4-6 bullet points identifying internal limitations: resource gaps, skill shortages, process inefficiencies, financial constraints. Be honest and specific
4. Opportunities (External, Positive) — 4-6 bullet points identifying external favorable factors: market trends, regulatory changes, technology shifts, underserved segments, partnerships
5. Threats (External, Negative) — 4-6 bullet points identifying external risks: competitive pressure, economic downturn, regulatory risk, technology disruption, supply chain issues
6. Strategic Actions — for each quadrant, list 1-2 specific action items:
   - Leverage: use Strengths to capture Opportunities
   - Defend: use Strengths to counter Threats
   - Improve: address Weaknesses to capture Opportunities
   - Mitigate: address Weaknesses to reduce Threats

Every bullet should be specific and actionable, not generic. Include evidence or reasoning for each point.""",
)

XLSX_RISK_ASSESSMENT = ArtifactTemplate(
    name="risk-assessment",
    description="Risk assessment, risk matrix, risk register, risk analysis, risk management plan",
    format="xlsx",
    category="business",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": False,
        "include_summary_row": False,
        "summary_type": "count",
    },
    context_prompt="""Create a risk assessment spreadsheet (risk register). Use these columns:
- Risk ID — unique identifier (R-001, R-002, etc.)
- Description — clear description of the risk event
- Category — type of risk (Technical, Financial, Operational, Legal, Reputational, Resource, Schedule)
- Probability — likelihood score 1-5 (1=Rare, 2=Unlikely, 3=Possible, 4=Likely, 5=Almost Certain)
- Impact — severity score 1-5 (1=Negligible, 2=Minor, 3=Moderate, 4=Major, 5=Critical)
- Risk Score — Probability multiplied by Impact (1-25)
- Mitigation Strategy — specific actions to reduce probability or impact
- Owner — person responsible for managing this risk
- Status — Open, Mitigating, Accepted, Closed
- Contingency Plan — what to do if the risk materializes despite mitigation

Sort by Risk Score descending (highest risks first). High risks (score 15-25) should be prominently flagged. Include a legend explaining the 1-5 scales for both Probability and Impact.""",
)

# ---------------------------------------------------------------------------
# PROJECT MANAGEMENT TEMPLATES (XLSX)
# ---------------------------------------------------------------------------

XLSX_GANTT_DATA = ArtifactTemplate(
    name="gantt-data",
    description="Gantt chart data, project timeline, project schedule, task timeline, milestone tracker, work breakdown",
    format="xlsx",
    category="business",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": True,
        "include_summary_row": False,
        "summary_type": "count",
    },
    context_prompt="""Create a Gantt chart data spreadsheet. Use these columns:
- Task — task name (indent sub-tasks with a prefix like "  " or use a WBS number like 1.1, 1.2)
- Start Date — task start date in YYYY-MM-DD format
- End Date — task end date in YYYY-MM-DD format
- Duration (days) — number of working days (End Date minus Start Date)
- % Complete — progress as decimal (0.0 to 1.0) for percentage formatting
- Dependencies — predecessor task references (e.g., "Task 2" or WBS number)
- Assigned To — person or team responsible
- Milestone — Y or N (milestones have zero duration: same start and end date)
- Status — Not Started, In Progress, Complete, Delayed

Group tasks by project phase (e.g., Phase 1: Planning, Phase 2: Design, Phase 3: Development, Phase 4: Testing, Phase 5: Deployment). Each phase should have a summary row. Milestones should mark key deliverables or decision points. Dependencies should form a logical sequence.""",
)

XLSX_STAKEHOLDER_MAP = ArtifactTemplate(
    name="stakeholder-map",
    description="Stakeholder map, RACI matrix, responsibility assignment, stakeholder analysis, project roles",
    format="xlsx",
    category="business",
    layout={
        "header_style": "bold_accent",
        "use_currency_format": False,
        "use_percentage_format": False,
        "include_summary_row": False,
        "summary_type": "count",
    },
    context_prompt="""Create a RACI matrix spreadsheet. Structure it as:
- First column: Task or Deliverable — list all major tasks, deliverables, or decisions (one per row)
- Subsequent columns: one column per stakeholder (person or role name in the header)
- Cell values: R (Responsible — does the work), A (Accountable — has final authority and approval), C (Consulted — provides input before the decision), I (Informed — notified after the decision)
- Rules: each row MUST have exactly one A. Each row should have at least one R. A person can be both R and A for the same task

Group rows by project phase or workstream. Include a legend at the top or bottom:
  R = Responsible (does the work)
  A = Accountable (approves, one per task)
  C = Consulted (provides input)
  I = Informed (kept in the loop)

Keep stakeholder columns to the key 5-10 roles. Use role titles rather than individual names when possible for reusability.""",
)


# ---------------------------------------------------------------------------
# Template Registry
# ---------------------------------------------------------------------------

ALL_TEMPLATES: list[ArtifactTemplate] = [
    # Presentations
    PPTX_CORPORATE_REPORT,
    PPTX_PITCH_DECK,
    PPTX_EDUCATIONAL,
    PPTX_TECHNICAL,
    # Documents
    PDF_BUSINESS_REPORT,
    PDF_RESEARCH_PAPER,
    PDF_PROJECT_PROPOSAL,
    PDF_TECHNICAL_DOC,
    # Spreadsheets
    XLSX_FINANCIAL,
    XLSX_PROJECT_TRACKER,
    XLSX_COMPARISON,
    # Charts
    CHART_TREND,
    CHART_COMPARISON,
    CHART_DISTRIBUTION,
    # Career
    PDF_RESUME,
    PDF_COVER_LETTER,
    # HR
    PDF_JOB_DESCRIPTION,
    PDF_OFFER_LETTER,
    PDF_PERFORMANCE_REVIEW,
    PDF_EMPLOYEE_HANDBOOK,
    PDF_TRAINING_MANUAL,
    # Legal
    PDF_NDA,
    PDF_SERVICE_AGREEMENT,
    # Operations
    PDF_MEETING_MINUTES,
    PDF_STATUS_REPORT,
    PDF_SOP,
    # Marketing
    PDF_CASE_STUDY,
    PDF_ONE_PAGER,
    PDF_NEWSLETTER,
    XLSX_CONTENT_CALENDAR,
    # Sales
    PDF_SALES_PROPOSAL,
    # Academic
    PDF_LESSON_PLAN,
    XLSX_RUBRIC,
    PDF_SYLLABUS,
    # Financial
    XLSX_INVOICE,
    XLSX_BUDGET_REPORT,
    XLSX_EXPENSE_REPORT,
    # Data/Analytics
    XLSX_KPI_DASHBOARD,
    XLSX_SURVEY_RESULTS,
    XLSX_INVENTORY_TRACKER,
    # Strategic
    PDF_SWOT_ANALYSIS,
    XLSX_RISK_ASSESSMENT,
    # Project Management
    XLSX_GANTT_DATA,
    XLSX_STAKEHOLDER_MAP,
]

# Index by format for quick filtering
TEMPLATES_BY_FORMAT: dict[str, list[ArtifactTemplate]] = {}
for _t in ALL_TEMPLATES:
    TEMPLATES_BY_FORMAT.setdefault(_t.format, []).append(_t)


def find_best_template(
    query: str,
    format: str = "",
    *,
    templates: list[ArtifactTemplate] | None = None,
) -> ArtifactTemplate | None:
    """Find the best matching template for a query using keyword matching.

    Falls back to simple keyword overlap scoring. For production use,
    this could be replaced with vector similarity via fastembed.

    Args:
        query: The user's request or document title
        format: Filter to specific format (pptx, pdf, xlsx, chart)
        templates: Override template list (for testing)

    Returns:
        Best matching template, or None if no good match
    """
    candidates = templates or ALL_TEMPLATES
    if format:
        candidates = [t for t in candidates if t.format == format]

    if not candidates:
        return None

    query_words = set(query.lower().split())

    best_score = 0
    best_template = None

    for template in candidates:
        # Score = number of query words found in template description
        desc_words = set(template.description.lower().split())
        name_words = set(template.name.lower().replace("-", " ").split())
        cat_words = set(template.category.lower().split())

        overlap = len(query_words & (desc_words | name_words | cat_words))

        # Bonus for exact category match
        if template.category.lower() in query.lower():
            overlap += 2

        if overlap > best_score:
            best_score = overlap
            best_template = template

    # Only return if we have a reasonable match (at least 1 word overlap)
    return best_template if best_score >= 1 else None


def get_template_context(template: ArtifactTemplate) -> str:
    """Get the context prompt to inject into the AI's system message.

    Returns a formatted string that tells the AI how to structure
    the artifact according to this template's design rules.
    """
    if not template:
        return ""

    parts = [f"[Template: {template.name}]"]
    parts.append(template.context_prompt)

    # Add layout-specific hints
    layout = template.layout
    if "slide_count_range" in layout:
        lo, hi = layout["slide_count_range"]
        parts.append(f"\nAim for {lo}-{hi} slides.")

    if "suggested_layouts" in layout:
        parts.append("\nSuggested slide structure:")
        for i, sl in enumerate(layout["suggested_layouts"], 1):
            parts.append(f"  {i}. [{sl['type']}] {sl['notes']}")

    rules = layout.get("design_rules", {})
    if rules:
        if rules.get("max_bullets_per_slide"):
            parts.append(f"\nMax {rules['max_bullets_per_slide']} bullets per slide.")
        if rules.get("max_words_per_bullet"):
            parts.append(f"Max {rules['max_words_per_bullet']} words per bullet point.")
        if rules.get("include_speaker_notes"):
            parts.append("Include detailed speaker notes for every slide.")

    return "\n".join(parts)


def get_template_for_tool_call(tool_name: str, user_message: str) -> str:
    """Match a template based on the tool being called and the user's request.

    Returns context prompt to inject, or empty string if no match.
    """
    format_map = {
        "create_document": "pdf",
        "create_presentation": "pptx",
        "create_spreadsheet": "xlsx",
        "create_chart": "chart",
    }
    fmt = format_map.get(tool_name, "")
    if not fmt:
        return ""

    template = find_best_template(user_message, format=fmt)
    if not template:
        return ""

    return get_template_context(template)


# Pipeline (build-mode) format → template format. The pipeline speaks in
# render formats (pdf/docx/pptx/xlsx/chart); templates are keyed by the same
# strings except docx shares the pdf document templates.
_PIPELINE_FORMAT_TO_TEMPLATE_FORMAT = {
    "pdf": "pdf",
    "docx": "pdf",
    "pptx": "pptx",
    "xlsx": "xlsx",
    "chart": "chart",
}


def get_pipeline_template_context(topic: str, pipeline_format: str) -> str:
    """Template context for the build pipeline's drafting prompts.

    The direct tool-call path injects templates via
    ``get_template_for_tool_call``; the build pipeline drafts content with
    its own prompts and historically skipped templates entirely — so all
    the "use specific numbers / 5-7 slices / sort by value" design guidance
    never reached build-mode output. This bridges that gap: it picks the
    best-matching template for the topic + format and returns its context
    prompt, or "" when nothing matches.
    """
    fmt = _PIPELINE_FORMAT_TO_TEMPLATE_FORMAT.get((pipeline_format or "").lower(), "")
    if not fmt:
        return ""
    template = find_best_template(topic, format=fmt)
    if not template:
        return ""
    return get_template_context(template)
