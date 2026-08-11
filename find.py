# find.py

import asyncio
import csv
import json
import re
from pathlib import Path

from browser_use import Agent, ChatOpenAI


# ============================================================
# CONFIG
# ============================================================

COMPANIES_FILE = "companies.json"
OUTPUT_FILE = "jobs.csv"

LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "unsloth/Qwen3.5-9B-GGUF:Q6_K"

JOB_SEARCH = "software engineer"

USE_VISION = False
MAX_ACTIONS_PER_STEP = 5


# ============================================================
# LOCAL QWEN
# ============================================================

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
)


# ============================================================
# LOAD COMPANIES
# ============================================================

with open(
    COMPANIES_FILE,
    "r",
    encoding="utf-8",
) as f:
    companies = json.load(f)


# ============================================================
# EXTRACT JSON LIST
# ============================================================

def extract_json(text):
    """
    Expected format ONLY:

    [
        {
            "title": "Software Engineer",
            "location": "Singapore",
            "job_url": "https://example.com/job/123"
        }
    ]
    """

    if not text:
        return []

    text = str(text).strip()

    # Remove markdown fences if the model adds them.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```\s*$",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------------------
    # Find JSON array
    # --------------------------------------------------------

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        return []

    try:

        data = json.loads(
            text[start:end + 1]
        )

    except json.JSONDecodeError as e:

        print(
            f"[JSON ERROR] {e}"
        )

        return []

    # ONLY accept a list.
    if not isinstance(data, list):

        print(
            "[JSON ERROR] Expected a list"
        )

        return []

    return data


# ============================================================
# BUILD TASK
# ============================================================

def build_task(company):

    name = company["company"]
    url = company["careers_link"]
    region = company.get("region", "")

    return f"""
Open this careers website:

{url}

Company:
{name}

Region:
{region}

SEARCH FOR:

"{JOB_SEARCH}"

Your ONLY task is to find ALL job listings matching:

"{JOB_SEARCH}"

============================================================
PROCESS
============================================================

1. Open the careers website.

2. Find the job search functionality.

3. Search for:
   "{JOB_SEARCH}"

4. If there is a search box, enter the search term.

5. If there are filters, use the search/filter.

6. Collect every job shown in the results.

7. Navigate through ALL result pages.

8. Click:
   - Next
   - pagination
   - Load more
   - Show more
   - View more

   whenever necessary.

9. Continue until there are no more results.

10. Do NOT open individual job detail pages.

============================================================
EXTRACT
============================================================

For every job listing collect ONLY:

- title
- location
- job_url

The job_url must be the URL of the individual job.

Do not invent URLs.

============================================================
IMPORTANT
============================================================

Do NOT:

- apply for jobs
- open individual job pages
- fill forms
- upload files
- submit applications
- collect descriptions
- evaluate candidates
- evaluate job suitability
- create files
- create results.md
- explain your work

Return ALL matching jobs.

Do NOT return only the first page.

Do NOT return only the first 10 jobs.

============================================================
OUTPUT
============================================================

Your final response MUST be ONLY a JSON LIST.

Example:

[
  {{
    "title": "Software Engineer",
    "location": "Singapore",
    "job_url": "https://example.com/job/123"
  }},
  {{
    "title": "Senior Software Engineer",
    "location": "Singapore",
    "job_url": "https://example.com/job/456"
  }}
]

The top-level JSON value MUST be a list.

No object wrapper.

No markdown.

No explanation.

Return ONLY the JSON list.
"""


# ============================================================
# FIND COMPANY
# ============================================================

async def find_company(company):

    name = company["company"]
    region = company.get("region", "")

    print()
    print("=" * 80)
    print(f"COMPANY : {name}")
    print(f"SEARCH  : {JOB_SEARCH}")
    print(f"REGION  : {region}")
    print(
        f"CAREERS : {company['careers_link']}"
    )
    print("=" * 80)

    task = build_task(company)

    try:

        agent = Agent(
            task=task,
            llm=llm,

            # Faster local inference.
            use_vision=USE_VISION,

            # Multiple browser actions per model call.
            max_actions_per_step=MAX_ACTIONS_PER_STEP,
        )

        history = await agent.run()

        result = history.final_result()

        print()
        print("AGENT RESULT:")
        print(result)
        print()

        jobs = extract_json(result)

        output = []

        for job in jobs:

            if not isinstance(job, dict):
                continue

            title = str(
                job.get("title", "")
            ).strip()

            location = str(
                job.get("location", "")
            ).strip()

            job_url = str(
                job.get("job_url", "")
            ).strip()

            if not title:
                continue

            if not job_url:
                continue

            output.append({
                "company": name,
                "region": region,
                "title": title,
                "location": location,
                "job_url": job_url,
            })

        # ----------------------------------------------------
        # Deduplicate company results
        # ----------------------------------------------------

        unique = {}

        for job in output:

            key = job["job_url"].rstrip("/")

            unique[key] = job

        output = list(
            unique.values()
        )

        print(
            f"Found {len(output)} jobs"
        )

        return output

    except Exception as e:

        print(
            f"[ERROR] {name}: {e}"
        )

        return []


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(jobs):

    fields = [
        "company",
        "region",
        "title",
        "location",
        "job_url",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(jobs)


# ============================================================
# MAIN
# ============================================================

async def main():

    print()
    print("=" * 80)
    print("JOB DISCOVERY")
    print("=" * 80)

    print(
        f"Companies : {len(companies)}"
    )

    print(
        f"Search    : {JOB_SEARCH}"
    )

    print(
        f"Output    : {Path(OUTPUT_FILE).absolute()}"
    )

    print("=" * 80)

    all_jobs = []

    for index, company in enumerate(
        companies,
        start=1,
    ):

        print(
            f"\n[{index}/{len(companies)}]"
        )

        if not company.get(
            "careers_link"
        ):

            print(
                "[SKIP] No careers_link"
            )

            continue

        if company.get(
            "careers_link"
        ) == "internal_path":

            print(
                "[SKIP] internal_path"
            )

            continue

        jobs = await find_company(
            company
        )

        all_jobs.extend(jobs)

        # ----------------------------------------------------
        # Global deduplication
        # ----------------------------------------------------

        unique = {}

        for job in all_jobs:

            key = (
                job["company"],
                job["job_url"].rstrip("/")
            )

            unique[key] = job

        all_jobs = list(
            unique.values()
        )

        # ----------------------------------------------------
        # Save after every company
        # ----------------------------------------------------

        save_csv(
            all_jobs
        )

        print(
            f"Total jobs: {len(all_jobs)}"
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Total jobs: {len(all_jobs)}"
    )

    print(
        f"CSV: {Path(OUTPUT_FILE).absolute()}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())