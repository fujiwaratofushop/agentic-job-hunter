import json
from browser_use import Agent, ChatOpenAI
import asyncio

JOBS_FILE = "jobs.json"
PROFILE_FILE = "profile.json"
RESUME_PATH = "Shirsak Sahoo – Resume.pdf"

with open(JOBS_FILE, "r") as f:
    jobs = json.load(f)

with open(PROFILE_FILE, "r") as f:
    profile = json.load(f)
profile_json_str = json.dumps(profile, indent=2)

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="unsloth/Qwen3.5-9B-GGUF:Q6_K",
)

print("=" * 80)
print("JOB APPLICATION AUTOMATION")
print("=" * 80)
print(f"Profile: {profile['name']} ({profile['role']})")
print(f"Experience: {profile['experience_years']} years")
print(f"Skills: {', '.join(profile['skills'])}")
print(f"Resume: {RESUME_PATH}")
print("=" * 80)

jobs_to_run = []
for job in jobs:
    if job.get("careers_link") == "internal_path":
        print(f"  [SKIP] {job['company']} ({job['region']}) - internal path")
        continue
    jobs_to_run.append(job)

print(f"\nFound {len(jobs_to_run)} jobs to process\n")
print("=" * 80)

for i, job in enumerate(jobs_to_run, 1):
    print(f"\n[{i}/{len(jobs_to_run)}] {job['company']} ({job['region']})")
    print(f"  Link: {job['careers_link']}")
    print(f"  In pipeline: {job.get('in_pipeline', False)}")
    print("-" * 80)

    task = f"Go to {job['careers_link']}. Fill form using profile: {profile_json_str}. Upload resume from {RESUME_PATH}."

    try:
        agent = Agent(task=task, llm=llm)
        asyncio.run(agent.run())
        print(f"\n[OK] Completed: {job['company']}")
    except Exception as e:
        print(f"\n[ERROR] {e}")

print("\n" + "=" * 80)
print("All jobs processed!")
print("=" * 80)

def main():
    print("=" * 80)
    print("JOB APPLICATION AUTOMATION")
    print("=" * 80)
    print(f"Profile: {profile['name']} ({profile['role']})")
    print(f"Experience: {profile['experience_years']} years")
    print(f"Skills: {', '.join(profile['skills'])}")
    print(f"Resume: {RESUME_PATH}")
    print("=" * 80)

    jobs_to_run = []
    for job in jobs:
        if job.get("careers_link") == "internal_path":
            print(f"  [SKIP] {job['company']} ({job['region']}) - internal path")
            continue
        jobs_to_run.append(job)

    print(f"\nFound {len(jobs_to_run)} jobs to process\n")
    print("=" * 80)

    for i, job in enumerate(jobs_to_run, 1):
        print(f"\n[{i}/{len(jobs_to_run)}] {job['company']} ({job['region']})")
        print(f"  Link: {job['careers_link']}")
        print(f"  In pipeline: {job.get('in_pipeline', False)}")
        print("-" * 80)

        task = f"Go to {job['careers_link']}. Fill form using profile: {profile_json_str}. Upload resume from {RESUME_PATH}."

        try:
            agent = Agent(task=task, llm=llm)
            asyncio.run(agent.run())
            print(f"\n[OK] Completed: {job['company']}")
        except Exception as e:
            print(f"\n[ERROR] {e}")

    print("\n" + "=" * 80)
    print("All jobs processed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
