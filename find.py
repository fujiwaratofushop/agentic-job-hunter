import asyncio
import csv
import json
import re
from pathlib import Path
from urllib.parse import urljoin

from browser_use import Agent, Controller, ActionResult, ChatOpenAI
from browser_use.browser.session import BrowserSession
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

COMPANIES_FILE = "companies.json"
OUTPUT_FILE = "jobs.csv"
SCRATCH_DIR = "scratch"  # per-company raw dumps written by the agent live here

LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "unsloth/Qwen3.5-9B-GGUF:Q6_K"

JOB_SEARCH = "software engineer"

USE_VISION = False
MAX_ACTIONS_PER_STEP = 5

# If True:
# "Software Engineer I - Backend" -> accepted
# "Principal Software Engineer"    -> accepted
# "Data Engineer"                 -> rejected
# "Machine Learning Engineer"     -> rejected
#
# This is intentionally strict because the requested search is
# "software engineer".
STRICT_TITLE_MATCH = True

# If a company returns more than this many raw rows but the accepted
# fraction is below RECALL_WARN_MIN_RATIO, print a loud warning - it
# almost always means the agent never used the site's search box and
# instead scraped (part of) the entire unfiltered job board.
RECALL_WARN_MIN_RAW = 60
RECALL_WARN_MIN_RATIO = 0.08


# ============================================================
# LOCAL LLM
# ============================================================

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model=LLM_MODEL,
)


# ============================================================
# COMPANIES
# ============================================================

def load_companies():
    with open(
        COMPANIES_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_companies(companies):
    temp_file = f"{COMPANIES_FILE}.tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            companies,
            f,
            indent=2,
            ensure_ascii=False,
        )

    Path(temp_file).replace(
        COMPANIES_FILE
    )


def save_company_recipe(
    companies,
    company,
    recipe,
):
    company["playwright"] = recipe

    save_companies(
        companies
    )

    print(
        f"[SAVED] Playwright recipe -> "
        f"{COMPANIES_FILE} -> "
        f"{company['company']}"
    )


# ============================================================
# SCRATCH FILES - incremental disk storage for the LLM
# ============================================================
#
# The core problem: a sub-12GB local model cannot reliably hold 70
# job cards across 7 pages in its context AND still emit a perfectly
# formed JSON blob at the very end - it drifts into prose instead
# (exactly what happened with Booking.com). So instead of asking it
# to "remember everything, then report", we give it a tool that
# writes straight to disk after every page. It never has to recall
# that data again, and the final step is just "call done()".

def ensure_scratch_dir():
    Path(SCRATCH_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )


def company_slug(company_name):
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        company_name.lower(),
    ).strip("-")

    return slug or "company"


def scratch_jobs_path(company):
    return Path(SCRATCH_DIR) / f"{company_slug(company['company'])}.jobs.jsonl"


def scratch_recipe_path(company):
    return Path(SCRATCH_DIR) / f"{company_slug(company['company'])}.recipe.json"


def reset_scratch_files(company):
    """
    Wipe the previous run's scratch files before a fresh browser-use
    attempt, so a failed/partial prior run can't leak stale data in.
    """

    for path in (
        scratch_jobs_path(company),
        scratch_recipe_path(company),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_scratch_jobs(company):
    path = scratch_jobs_path(company)

    if not path.exists():
        return []

    jobs = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[SCRATCH] Skipping unparsable line "
                    f"{line_number} in {path}"
                )

    return jobs


def read_scratch_recipe(company):
    path = scratch_recipe_path(company)

    if not path.exists():
        return None

    raw = path.read_text(encoding="utf-8").strip()

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(
            f"[SCRATCH] Recipe in {path} is not valid JSON, ignoring"
        )
        return None


# ============================================================
# CONTROLLER - tools the agent calls to save raw data as it goes
# ============================================================
#
# These are deliberately "dumb": the agent is NOT asked to judge
# relevance here. It just copies what's on the page. All filtering
# (title match, region match) happens afterwards in normalize_jobs(),
# using the exact same logic every time - not a fresh LLM judgment
# call per page that can drift or miss things.

class JobItem(BaseModel):
    title: str = ""
    location: str = ""
    job_url: str = ""


class ExtractJobsParams(BaseModel):
    card_selector: str = Field(..., description="CSS selector for job cards (e.g. '.job-card', 'article.job-listing')")
    title_selector: str = Field(..., description="CSS selector for job titles within each card")
    location_selector: str = Field(..., description="CSS selector for job locations within each card")
    url_selector: str = Field(..., description="CSS selector for job URLs within each card")


class SaveRecipeParams(BaseModel):
    recipe_json: str


def build_controller(company):

    jobs_path = scratch_jobs_path(company)
    recipe_path = scratch_recipe_path(company)

    controller = Controller()

    @controller.action(
        "Extract every job card on the CURRENT page using CSS selectors and save "
        "to disk. You provide the SELECTORS (e.g. '.job-card', 'a.job-title'), "
        "NOT the job data itself - the tool reads the page and copies exact "
        "title/location/href for every matching element. Call this once per "
        "page, before pagination. The tool will read the DOM and extract data "
        "deterministically - you do NOT type out the job data. If 0 cards match, "
        "your selector is probably wrong - try again with a different selector.",
        param_model=ExtractJobsParams,
    )
    async def extract_jobs_by_selector(
        params: ExtractJobsParams,
        browser_session: BrowserSession,
    ) -> ActionResult:
        page = await browser_session.must_get_current_page()

        jobs = await extract_cards(page, {
            "card_selector": params.card_selector,
            "title_selector": params.title_selector,
            "location_selector": params.location_selector,
            "url_selector": params.url_selector,
        })

        if not jobs:
            return ActionResult(
                extracted_content=f"0 cards matched '{params.card_selector}' - selector is probably wrong, try again.",
                include_in_memory=True,  # let it see the failure and retry
            )

        with open(jobs_path, "a", encoding="utf-8") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")

        return ActionResult(
            extracted_content=f"Extracted {len(jobs)} jobs via selector, saved to disk.",
            include_in_memory=False,
        )

    @controller.action(
        "Save the Playwright recipe (the steps needed to reproduce this "
        "search and pagination from scratch) to disk. Call this ONCE, "
        "whenever you've worked out reliable selectors - it's fine to "
        "call it early and it will be overwritten if you call it again "
        "later with better selectors. Pass a JSON string matching the "
        "schema you were given in the task.",
        param_model=SaveRecipeParams,
    )
    def save_playwright_recipe(params: SaveRecipeParams) -> ActionResult:

        recipe_path.write_text(
            params.recipe_json,
            encoding="utf-8",
        )

        return ActionResult(
            extracted_content="Recipe saved to disk.",
            include_in_memory=False,
        )

    return controller


# ============================================================
# REGION
# ============================================================

def get_regions(company):
    regions = company.get(
        "region",
        [],
    )

    if isinstance(
        regions,
        str,
    ):
        regions = [regions]

    return [
        str(region).strip()
        for region in regions
        if str(region).strip()
    ]


def location_matches_region(
    location,
    regions,
):
    """
    Region matching uses word boundaries so short region names don't
    false-positive on substrings of longer place names, e.g. region
    "India" must NOT match location "Indianapolis, IN, USA", and
    region "Georgia" (country) must NOT match "Georgia, USA" being
    conflated across country/state - callers still get the broad
    "Amsterdam, North Holland, Netherlands" matches "Netherlands"
    behavior, just without the accidental substring hits.
    """

    if not location:
        return False

    location_lower = location.lower()

    for region in regions:

        region_lower = region.lower().strip()

        if not region_lower:
            continue

        pattern = r"\b" + re.escape(region_lower) + r"\b"

        if re.search(pattern, location_lower):
            return True

    return False


# ============================================================
# TITLE FILTER
# ============================================================

def _build_title_pattern(search_term):
    """
    Word-boundary regex instead of a naive substring check.

    Naive substring matching has a real bug: "software engineer" is a
    literal prefix of "software engineering", so a title like
    "Software Engineering Manager" or "Senior Software Engineering
    Lead" would incorrectly pass title_matches_search() under plain
    `search in normalized_title`. \\b after "engineer" fixes this,
    since there's no word boundary between "engineer" and "ing".
    """

    words = [
        re.escape(word)
        for word in search_term.lower().split()
    ]

    if not words:
        return re.compile(r"(?!x)x")  # matches nothing

    return re.compile(
        r"\b" + r"\s+".join(words) + r"\b"
    )


_TITLE_PATTERN = _build_title_pattern(JOB_SEARCH)


def title_matches_search(title):
    """
    Enforce the requested job title in Python.

    This prevents the LLM from returning unrelated search
    results such as:

        Data Engineer
        Solutions Architect
        Product Manager
        Director Engineering
        Machine Learning Engineer
        Software Engineering Manager   <- see _build_title_pattern

    when the requested search is:

        software engineer
    """

    if not title:
        return False

    normalized_title = (
        title
        .lower()
        .strip()
    )

    if STRICT_TITLE_MATCH:
        return bool(_TITLE_PATTERN.search(normalized_title))

    # Future looser matching can go here.
    return bool(_TITLE_PATTERN.search(normalized_title))


# ============================================================
# URL NORMALIZATION
# ============================================================

def clean_url(url):
    """
    Convert relative URLs into absolute URLs.

    Also removes accidental Markdown link syntax.
    """

    if not url:
        return ""

    url = str(url).strip()

    # Markdown:
    #
    # [https://example.com](https://example.com)
    #
    markdown_match = re.search(
        r"\]\((https?://[^)]+)\)",
        url,
    )

    if markdown_match:
        url = markdown_match.group(1)

    # Remove surrounding markdown/code characters.
    url = url.strip(
        "`'\" "
    )

    return url


def make_absolute_url(
    url,
    base_url,
):
    url = clean_url(
        url
    )

    if not url:
        return ""

    return urljoin(
        base_url,
        url,
    )


# ============================================================
# RECIPE NORMALIZATION
# ============================================================

def normalize_selector(selector):
    """
    Fix common invalid selectors produced by an LLM.

    Example:

        mat-panel-title a#link-job-*

    is NOT valid CSS.

    Convert to:

        mat-panel-title a[id^="link-job-"]
    """

    if not selector:
        return selector

    selector = str(
        selector
    ).strip()

    # Common wildcard ID pattern.
    selector = re.sub(
        r"#([A-Za-z0-9_-]+)-\*",
        r'[id^="\1-"]',
        selector,
    )

    # Specific Booking.com pattern.
    selector = selector.replace(
        "a#link-job-*",
        'a[id^="link-job-"]',
    )

    return selector


def normalize_recipe(
    recipe,
    company,
):
    """
    Normalize the LLM-generated recipe before saving it.

    This means companies.json contains the corrected,
    reusable recipe rather than the raw LLM output.
    """

    if not isinstance(
        recipe,
        dict,
    ):
        return None

    recipe = json.loads(
        json.dumps(recipe)
    )

    steps = recipe.get(
        "steps"
    )

    if not isinstance(
        steps,
        list,
    ):
        return None

    base_url = company[
        "careers_link"
    ]

    for step in steps:

        if not isinstance(
            step,
            dict,
        ):
            continue

        action = step.get(
            "action"
        )

        # Normalize URLs.
        if action == "goto":

            if step.get("url"):
                step["url"] = make_absolute_url(
                    step["url"],
                    base_url,
                )

        # Normalize selectors.
        for key in (
            "selector",
            "card_selector",
            "title_selector",
            "location_selector",
            "url_selector",
            "next_selector",
        ):

            if key in step:
                step[key] = normalize_selector(
                    step[key]
                )

    recipe["version"] = int(
        recipe.get(
            "version",
            1,
        )
    )

    return recipe


# ============================================================
# EXTRACT JSON FROM AGENT RESULT (fallback only - see below)
# ============================================================

def extract_agent_result(text):
    """
    Fallback text parser, used ONLY if the agent never called the
    save_jobs tool during the run (e.g. controller misconfigured, or
    an older browser-use version). The primary path is now reading
    the scratch files written by save_jobs / save_playwright_recipe -
    see read_scratch_jobs() / read_scratch_recipe() and
    learn_with_browser_use().

    Browser-use may return:

        Here is the result:

        **JOBS**

        ...

        **PLAYWRIGHT RECIPE**
        {
            ...
        }

    So do NOT assume the whole response is JSON.

    Search the response for a JSON object containing
    "jobs" and/or "playwright".
    """

    if not text:
        return None

    text = str(
        text
    ).strip()

    # --------------------------------------------------------
    # First attempt: entire response is JSON.
    # --------------------------------------------------------

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```\s*$",
        "",
        cleaned,
    )

    try:

        data = json.loads(
            cleaned
        )

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception:
        pass

    # --------------------------------------------------------
    # Second attempt:
    # Find JSON object containing "jobs".
    # --------------------------------------------------------

    decoder = json.JSONDecoder()

    for match in re.finditer(
        r"\{",
        text,
    ):

        start = match.start()

        try:

            data, end = decoder.raw_decode(
                text[start:]
            )

        except json.JSONDecodeError:
            continue

        if not isinstance(
            data,
            dict,
        ):
            continue

        if (
            "jobs" in data
            or "playwright" in data
        ):
            return data

    # --------------------------------------------------------
    # Third attempt:
    # Extract PLAYWRIGHT RECIPE specifically.
    # --------------------------------------------------------

    recipe_match = re.search(
        r"PLAYWRIGHT\s+RECIPE\s*:?\s*",
        text,
        flags=re.IGNORECASE,
    )

    if recipe_match:

        start = text.find(
            "{",
            recipe_match.end(),
        )

        if start != -1:

            try:

                data, _ = decoder.raw_decode(
                    text[start:]
                )

                if isinstance(
                    data,
                    dict,
                ):
                    return {
                        "jobs": [],
                        "playwright": data,
                    }

            except json.JSONDecodeError:
                pass

    print(
        "[JSON ERROR] "
        "Could not extract structured data "
        "from browser-use result"
    )

    return None


# ============================================================
# BROWSER-USE TASK
# ============================================================

def build_task(company):

    name = company[
        "company"
    ]

    careers_url = company[
        "careers_link"
    ]

    regions = get_regions(
        company
    )

    return f"""
Open this careers website:

{careers_url}

Company:
{name}

TARGET REGION (context only, to help you use the site's own search /
location filter - do NOT use this to filter results yourself):
{json.dumps(regions, ensure_ascii=False)}

SEARCH TERM (context only, to help you use the site's own search box -
do NOT use this to filter results yourself):
{JOB_SEARCH}

============================================================
YOUR ONLY JOB: COLLECT EVERYTHING, DO NOT FILTER
============================================================

Do NOT decide which jobs are relevant. Do NOT skip a job because its
title looks wrong. Do NOT judge location. All of that filtering
happens afterwards in code, using the raw data you save - it is not
your job to judge it.

Steps:

1. MANDATORY FIRST STEP - do this before anything else in step 2 below:
    use the site's own search box, filter dropdown, or category link to
    narrow results down to "{JOB_SEARCH}"{f" and, if available, {regions[0]}" if regions else ""}.
    Do NOT skip this and do NOT fall back to browsing the full, unfiltered
    job list just because a search box is hard to find. If you cannot
    locate a text search box within a few actions, look instead for a
    department/category/team filter (e.g. "Engineering") and use that as a
    narrower substitute before moving on.


2. On EVERY page of results, call the extract_jobs_by_selector tool ONCE with every
   single job card visible on that page - title, location, and link,
   exactly as shown. Copy ALL of them, even ones that look unrelated
   to "{JOB_SEARCH}". Do not summarize, do not paraphrase the title.
   Provide the CSS selectors for cards, titles, locations, and URLs.
   The tool will read the DOM and extract data deterministically.

3. After calling extract_jobs_by_selector for a page, click Next / pagination /
   Load more / Show more / View more to reach the next page, and
   repeat step 2. Keep going until there are no more results.

4. While you're doing this, work out the Playwright steps needed to
   reproduce the search from a fresh page load (goto, fill, click,
   wait, extract_and_paginate with stable CSS selectors). Once you're
   confident in them, call save_playwright_recipe with that JSON as a
   string. For wildcard ids use [id^="prefix-"], never "#prefix-*"
   (that is not valid CSS).

5. Once there are no more pages, call `done` with success=true and a
   short one-line message, e.g. "Saved 7 pages of results." Do NOT try
   to compile, list, or repeat the job data in your final answer - it
   is already saved to disk via extract_jobs_by_selector.

============================================================
PLAYWRIGHT RECIPE JSON SHAPE
============================================================

Pass this as a JSON string to save_playwright_recipe:

{{
  "version": 1,
  "steps": [
    {{"action": "goto", "url": "..."}},
    {{"action": "fill", "selector": "...", "value": "..."}},
    {{"action": "click", "selector": "..."}},
    {{"action": "wait", "selector": "..."}},
    {{
      "action": "extract_and_paginate",
      "card_selector": "...",
      "title_selector": "...",
      "location_selector": "...",
      "url_selector": "...",
      "next_selector": "..."
    }}
  ]
}}
"""


# ============================================================
# JOB NORMALIZATION
# ============================================================

def normalize_jobs(
    company,
    jobs,
):
    name = company[
        "company"
    ]

    regions = get_regions(
        company
    )

    output = []

    for job in jobs or []:

        if not isinstance(
            job,
            dict,
        ):
            continue

        title = str(
            job.get(
                "title",
                "",
            )
        ).strip()

        location = str(
            job.get(
                "location",
                "",
            )
        ).strip()

        job_url = str(
            job.get(
                "job_url",
                "",
            )
        ).strip()

        # ----------------------------------------------------
        # HARD TITLE FILTER
        # ----------------------------------------------------

        if not title_matches_search(
            title
        ):
            print(
                f"[FILTER] "
                f"Rejected title: "
                f"{title}"
            )

            continue

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        job_url = make_absolute_url(
            job_url,
            company[
                "careers_link"
            ],
        )

        if not job_url:
            print(
                f"[FILTER] "
                f"Rejected empty URL: "
                f"{title} @ {location}"
            )
            continue

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        if not location_matches_region(
            location,
            regions,
        ):

            print(
                f"[FILTER] "
                f"Rejected location: "
                f"{location}"
            )

            continue

        # Always normalize output location
        # to the configured region.
        normalized_location = (
            regions[0]
            if regions
            else location
        )

        output.append({
            "company": name,
            "region": ", ".join(
                regions
            ),
            "title": title,
            "location": normalized_location,
            "job_url": job_url,
        })

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    # Use (title, location, job_url) as the dedup key instead of just job_url
    # This prevents two genuinely different postings that both fail to get a url
    # from being silently merged into one row.
    unique = {}

    for job in output:

        # Log when job_url is empty before dropping it
        if not job_url:
            print(
                f"[DUP/EMPTY] "
                f"Job with empty URL, skipping dedup: "
                f"{title} @ {location}"
            )
            continue

        key = (
            job[
                "job_url"
            ]
            .rstrip("/")
        )

        unique[key] = job

    return list(
        unique.values()
    )


# ============================================================
# PLAYWRIGHT EXTRACTION
# ============================================================

async def extract_cards(
    page,
    step,
):
    cards = page.locator(
        step[
            "card_selector"
        ]
    )

    count = await cards.count()

    jobs = []

    for index in range(
        count
    ):

        card = cards.nth(
            index
        )

        try:

            title = await card.locator(
                step[
                    "title_selector"
                ]
            ).inner_text()

        except Exception:

            title = ""

        try:

            location = await card.locator(
                step[
                    "location_selector"
                ]
            ).inner_text()

        except Exception:

            location = ""

        try:

            href = await card.locator(
                step[
                    "url_selector"
                ]
            ).get_attribute(
                "href"
            )

        except Exception:

            href = None

        if not href:
            continue

        job_url = make_absolute_url(
            href,
            page.url,
        )

        jobs.append({
            "title": title.strip(),
            "location": location.strip(),
            "job_url": job_url,
        })

    return jobs


async def execute_playwright_recipe(
    company,
):
    recipe = company.get(
        "playwright"
    )

    if not isinstance(
        recipe,
        dict,
    ):
        return None

    steps = recipe.get(
        "steps"
    )

    if not isinstance(
        steps,
        list,
    ) or not steps:

        return None

    # Normalize the saved recipe before using it.
    recipe = normalize_recipe(
        recipe,
        company,
    )

    print(
        f"[PLAYWRIGHT] "
        f"Using saved recipe for "
        f"{company['company']}"
    )

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=True
        )

        page = await browser.new_page()

        try:

            all_jobs = []

            for step in recipe[
                "steps"
            ]:

                if not isinstance(
                    step,
                    dict,
                ):
                    continue

                action = step.get(
                    "action"
                )

                # ------------------------------------------------
                # GOTO
                # ------------------------------------------------

                if action == "goto":

                    await page.goto(
                        step[
                            "url"
                        ],
                        wait_until=(
                            "domcontentloaded"
                        ),
                        timeout=30000,
                    )

                # ------------------------------------------------
                # FILL
                # ------------------------------------------------

                elif action == "fill":

                    await page.locator(
                        step[
                            "selector"
                        ]
                    ).fill(
                        step[
                            "value"
                        ]
                    )

                # ------------------------------------------------
                # CLICK
                # ------------------------------------------------

                elif action == "click":

                    await page.locator(
                        step[
                            "selector"
                        ]
                    ).click()

                # ------------------------------------------------
                # WAIT
                # ------------------------------------------------

                elif action == "wait":

                    await page.locator(
                        step[
                            "selector"
                        ]
                    ).first.wait_for(
                        state="visible",
                        timeout=15000,
                    )

                # ------------------------------------------------
                # EXTRACT + PAGINATE
                # ------------------------------------------------

                elif (
                    action
                    == "extract_and_paginate"
                ):

                    max_pages = int(
                        step.get(
                            "max_pages",
                            500,
                        )
                    )

                    seen_signatures = set()

                    for page_number in range(
                        1,
                        max_pages + 1,
                    ):

                        print(
                            f"[PLAYWRIGHT] "
                            f"{company['company']} "
                            f"page {page_number}"
                        )

                        await page.wait_for_timeout(
                            step.get(
                                "wait_ms",
                                500,
                            )
                        )

                        page_jobs = (
                            await extract_cards(
                                page,
                                step,
                            )
                        )

                        if not page_jobs:

                            await page.wait_for_timeout(
                                1000
                            )

                            page_jobs = (
                                await extract_cards(
                                    page,
                                    step,
                                )
                            )

                        all_jobs.extend(
                            page_jobs
                        )

                        urls = [
                            job[
                                "job_url"
                            ]
                            for job in page_jobs
                            if job.get(
                                "job_url"
                            )
                        ]

                        signature = (
                            page.url,
                            tuple(
                                urls[:3]
                            ),
                            tuple(
                                urls[-3:]
                            ),
                        )

                        if (
                            signature
                            in seen_signatures
                        ):

                            print(
                                "[PLAYWRIGHT] "
                                "Pagination loop detected"
                            )

                            break

                        seen_signatures.add(
                            signature
                        )

                        next_selector = (
                            step.get(
                                "next_selector"
                            )
                        )

                        if not next_selector:
                            break

                        next_button = (
                            page.locator(
                                next_selector
                            ).first
                        )

                        if (
                            await next_button.count()
                            == 0
                        ):
                            break

                        try:

                            if not await next_button.is_visible():
                                break

                        except Exception:

                            break

                        try:

                            if await next_button.is_disabled():
                                break

                        except Exception:
                            pass

                        try:

                            aria_disabled = (
                                await next_button.get_attribute(
                                    "aria-disabled"
                                )
                            )

                            if (
                                aria_disabled
                                == "true"
                            ):
                                break

                        except Exception:
                            pass

                        old_url = page.url

                        old_first_url = (
                            urls[0]
                            if urls
                            else ""
                        )

                        try:

                            await next_button.click(
                                timeout=5000
                            )

                        except Exception:

                            break

                        try:

                            await page.wait_for_load_state(
                                "domcontentloaded",
                                timeout=5000,
                            )

                        except Exception:
                            pass

                        await page.wait_for_timeout(
                            step.get(
                                "after_next_wait_ms",
                                800,
                            )
                        )

                        changed = False

                        if page.url != old_url:

                            changed = True

                        else:

                            new_jobs = (
                                await extract_cards(
                                    page,
                                    step,
                                )
                            )

                            new_urls = [
                                job[
                                    "job_url"
                                ]
                                for job in new_jobs
                                if job.get(
                                    "job_url"
                                )
                            ]

                            new_first_url = (
                                new_urls[0]
                                if new_urls
                                else ""
                            )

                            if (
                                new_first_url
                                and new_first_url
                                != old_first_url
                            ):

                                changed = True

                        if not changed:

                            print(
                                "[PLAYWRIGHT] "
                                "Next did not change results"
                            )

                            break

                else:

                    print(
                        f"[PLAYWRIGHT] "
                        f"Unknown step action "
                        f"'{action}', skipping"
                    )

            await browser.close()

            if not all_jobs:

                print(
                    "[PLAYWRIGHT] "
                    "No jobs extracted"
                )

                return None

            jobs = normalize_jobs(
                company,
                all_jobs,
            )

            print(
                f"[PLAYWRIGHT] "
                f"Accepted {len(jobs)} jobs"
            )

            return jobs

        except Exception as e:

            print(
                f"[PLAYWRIGHT FAILED] "
                f"{company['company']}: {e}"
            )

            await browser.close()

            return None


# ============================================================
# BROWSER-USE LEARNING
# ============================================================

async def learn_with_browser_use(
    companies,
    company,
):
    print(
        f"[BROWSER-USE] "
        f"Learning {company['company']}"
    )

    ensure_scratch_dir()
    reset_scratch_files(company)

    task = build_task(
        company
    )

    controller = build_controller(
        company
    )

    agent = Agent(
        task=task,
        llm=llm,
        controller=controller,
        use_vision=USE_VISION,
        max_actions_per_step=(
            MAX_ACTIONS_PER_STEP
        ),
    )

    history = await agent.run()

    result = (
        history.final_result()
    )

    print()
    print(
        "[BROWSER-USE RESULT]"
    )
    print(result)
    print()

    # --------------------------------------------------------
    # Primary path: read what the agent saved to disk as it went.
    # This is what we trust - not the free-text final answer, which
    # a sub-12GB model cannot reliably format as clean JSON once the
    # task has run for many steps.
    # --------------------------------------------------------

    raw_jobs = read_scratch_jobs(
        company
    )

    recipe = read_scratch_recipe(
        company
    )

    if raw_jobs:

        print(
            f"[SCRATCH] "
            f"Read {len(raw_jobs)} raw job rows from disk"
        )

    else:

        print(
            "[SCRATCH] "
            "No jobs were saved to disk via the tool"
        )

    # --------------------------------------------------------
    # Fallback: only if the model never called save_jobs at all,
    # fall back to the old text-parsing approach as a safety net.
    # --------------------------------------------------------

    if not raw_jobs:

        data = extract_agent_result(
            result
        )

        if data:

            raw_jobs = data.get(
                "jobs",
                [],
            )

            if recipe is None:

                fallback_recipe = data.get(
                    "playwright"
                )

                if isinstance(
                    fallback_recipe,
                    dict,
                ):
                    recipe = fallback_recipe

    # --------------------------------------------------------
    # Save recipe.
    #
    # Even if the agent's jobs are bad, the route is useful.
    # --------------------------------------------------------

    if isinstance(
        recipe,
        dict,
    ):

        recipe = normalize_recipe(
            recipe,
            company,
        )

        if recipe:

            save_company_recipe(
                companies,
                company,
                recipe,
            )

        else:

            print(
                "[BROWSER-USE] "
                "Recipe found but failed normalization"
            )

    else:

        print(
            "[BROWSER-USE] "
            "No Playwright recipe found"
        )

    # --------------------------------------------------------
    # Normalize/filter raw jobs. This is the ONLY place filtering
    # happens now - the agent no longer judges relevance itself.
    # --------------------------------------------------------

    jobs = normalize_jobs(
        company,
        raw_jobs,
    )

    print(
        f"[BROWSER-USE] "
        f"Accepted {len(jobs)} jobs "
        f"(after filtering {len(raw_jobs)} raw rows)"
    )

    # --------------------------------------------------------
    # Recall diagnostic: if there were plenty of raw rows but almost
    # none passed the title filter, the agent very likely never
    # applied the site's own search box and instead scraped (part
    # of) the full, unfiltered job board - which also means it may
    # have burned its action budget before reaching the pages that
    # actually contain matching roles. This does not affect what
    # gets written to the CSV (that's already correct), it's purely
    # a signal for you to go check that company.
    # --------------------------------------------------------

    if len(raw_jobs) >= RECALL_WARN_MIN_RAW:

        ratio = (
            len(jobs) / len(raw_jobs)
            if raw_jobs
            else 0
        )

        if ratio < RECALL_WARN_MIN_RATIO:

            print(
                f"[WARN] {company['company']}: {len(raw_jobs)} raw rows "
                f"scraped but only {len(jobs)} matched '{JOB_SEARCH}' "
                f"({ratio:.0%}). This usually means the agent browsed the "
                f"full unfiltered job list instead of using the site's "
                f"search box - check scratch/{company_slug(company['company'])}.jobs.jsonl "
                f"and consider adding this domain to _ATS_SEARCH_PARAM if "
                f"it's on a known ATS."
            )

    return jobs


# ============================================================
# FIND COMPANY
# ============================================================

async def find_company(
    companies,
    company,
):
    name = company[
        "company"
    ]

    regions = get_regions(
        company
    )

    print()
    print("=" * 80)
    print(
        f"COMPANY : {name}"
    )
    print(
        f"SEARCH  : {JOB_SEARCH}"
    )
    print(
        f"REGION  : {regions}"
    )
    print(
        f"CAREERS : "
        f"{company['careers_link']}"
    )
    print("=" * 80)

    # ========================================================
    # FAST PATH
    # ========================================================

    if company.get(
        "playwright"
    ):

        print(
            "[PLAYWRIGHT] "
            "Saved recipe found"
        )

        jobs = (
            await execute_playwright_recipe(
                company
            )
        )

        if jobs is not None:

            return jobs

        print(
            "[PLAYWRIGHT] "
            "Saved recipe failed"
        )

        print(
            "[BROWSER-USE] "
            "Re-learning route"
        )

    # ========================================================
    # SLOW PATH
    # ========================================================

    try:

        return await learn_with_browser_use(
            companies,
            company,
        )

    except Exception as e:

        print(
            f"[ERROR] {name}: {e}"
        )

        return []


# ============================================================
# CSV APPEND
# ============================================================

def append_csv(
    jobs,
):
    if not jobs:
        return

    fields = [
        "company",
        "region",
        "title",
        "location",
        "job_url",
    ]

    output_path = Path(
        OUTPUT_FILE
    )

    file_has_data = (
        output_path.exists()
        and output_path.stat().st_size > 0
    )

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        if not file_has_data:

            writer.writeheader()

        writer.writerows(
            jobs
        )

    print(
        f"[CSV] "
        f"Appended {len(jobs)} jobs"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    companies = load_companies()

    print()
    print("=" * 80)
    print(
        "JOB DISCOVERY"
    )
    print("=" * 80)

    print(
        f"Companies : "
        f"{len(companies)}"
    )

    print(
        f"Search    : "
        f"{JOB_SEARCH}"
    )

    print(
        f"CSV       : "
        f"{Path(OUTPUT_FILE).absolute()}"
    )

    print(
        f"Companies : "
        f"{Path(COMPANIES_FILE).absolute()}"
    )

    print("=" * 80)

    total_jobs_this_run = 0

    for index, company in enumerate(
        companies,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(companies)}]"
        )

        if not company.get(
            "careers_link"
        ):

            print(
                "[SKIP] "
                "No careers_link"
            )

            continue

        if (
            company.get(
                "careers_link"
            )
            == "internal_path"
        ):

            print(
                "[SKIP] "
                "internal_path"
            )

            continue

        jobs = await find_company(
            companies,
            company,
        )

        # Append. Never overwrite.
        append_csv(
            jobs
        )

        total_jobs_this_run += len(
            jobs
        )

        print(
            f"[COMPANY DONE] "
            f"{company['company']} -> "
            f"{len(jobs)} jobs"
        )

        print(
            f"[RUN TOTAL] "
            f"{total_jobs_this_run}"
        )

    print()
    print("=" * 80)
    print(
        "DONE"
    )
    print("=" * 80)

    print(
        f"Jobs appended this run: "
        f"{total_jobs_this_run}"
    )

    print(
        f"CSV: "
        f"{Path(OUTPUT_FILE).absolute()}"
    )

    print(
        f"Companies: "
        f"{Path(COMPANIES_FILE).absolute()}"
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())