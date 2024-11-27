import logging
import asyncio
from typing import Dict, Any, List, Optional
import aiohttp
import typer
import pandas as pd
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import Column, Integer, String, Boolean, Text, DateTime, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()


class JobPost(Base):
    """SQLAlchemy model for storing job post details"""
    __tablename__ = 'job_posts'

    id = Column(Integer, primary_key=True)
    job_vision_id = Column(Integer, unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    company = Column(String(255))
    description = Column(Text)
    is_remote = Column(Boolean, default=False)
    is_internship = Column(Boolean, default=False)
    salary_range = Column(Integer)
    work_experience = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AsyncJobVisionScraper:
    """Async web scraper for JobVision job posts"""
    BASE_URL = "https://candidateapi.jobvision.ir/api/v1/JobPost/List"
    JOB_DETAILS_URL = "https://candidateapi.jobvision.ir/api/v1/JobPost/Detail?jobPostId={jobPostId}"

    def __init__(self, database_path: str = 'job_posts.db'):
        """
        Initialize the async scraper with a SQLite database

        :param database_path: Path to the SQLite database file
        """
        # Create async database engine
        self.engine = create_async_engine(
            f'sqlite+aiosqlite:///{database_path}',
            echo=False
        )

        # Create an async session factory
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )

    def _create_payload(
            self,
            keyword: str = "backend",
            page: int = 1,
            page_size: int = 30
    ) -> Dict[str, Any]:
        """
        Create payload for job search API request

        :param keyword: Search keyword
        :param page: Page number
        :param page_size: Number of jobs per page
        :return: Payload dictionary
        """
        return {
            "pageSize": page_size,
            "requestedPage": page,
            "keyword": keyword,
            "sortBy": 1,
            "searchId": None
        }

    async def _extract_job_details(
            self,
            session: aiohttp.ClientSession,
            job_post: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Asynchronously extract detailed information for a job post

        :param session: Aiohttp client session
        :param job_post: Basic job post information
        :return: Detailed job post information or None
        """
        try:
            async with session.get(
                    self.JOB_DETAILS_URL.format(jobPostId=job_post["id"]),
                    timeout=10
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Error fetching job details for {job_post['id']}: {e}")
            return None

    async def _save_job_post(self, job_details: Dict[str, Any]):
        """
        Asynchronously save job post to database, avoiding duplicates

        :param job_details: Detailed job post information
        """
        async with self.AsyncSessionLocal() as session:
            try:
                # Check if job already exists using a query
                existing_job = await session.execute(
                    select(JobPost).filter_by(job_vision_id=job_details.get('id'))
                )
                existing_job = existing_job.scalar_one_or_none()

                if existing_job:
                    logger.info(f"Job {job_details.get('id')} already exists. Skipping.")
                    return

                # Create new job post
                new_job = JobPost(
                    job_vision_id=job_details.get('id'),
                    title=job_details.get('title', ''),
                    company=job_details.get('companyName', ''),
                    description=job_details.get('description', ''),
                    is_remote=job_details.get('isRemote', False),
                    is_internship=job_details.get('isInternship', False),
                    salary_range=job_details.get('salaryRangeId'),
                    work_experience=job_details.get('workExperienceId')
                )

                session.add(new_job)
                await session.commit()
                logger.info(f"Saved job: {new_job.title}")

            except IntegrityError:
                await session.rollback()
                logger.warning(f"Duplicate job entry: {job_details.get('id')}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving job: {e}")

    async def _process_job_posts(
            self,
            session: aiohttp.ClientSession,
            job_posts: List[Dict[str, Any]]
    ):
        """
        Asynchronously process a list of job posts

        :param session: Aiohttp client session
        :param job_posts: List of job posts to process
        """
        # Use asyncio.gather to process jobs concurrently
        tasks = []
        for job_post in job_posts:
            task = asyncio.create_task(self._process_single_job(session, job_post))
            tasks.append(task)

        await asyncio.gather(*tasks)

    async def _process_single_job(
            self,
            session: aiohttp.ClientSession,
            job_post: Dict[str, Any]
    ):
        """
        Process a single job post

        :param session: Aiohttp client session
        :param job_post: Job post to process
        """
        job_details = await self._extract_job_details(session, job_post)
        if job_details and 'data' in job_details:
            await self._save_job_post(job_details['data'])

    async def scrape_jobs(
            self,
            keyword: str = "backend",
    ):
        """
        Asynchronously scrape job posts from JobVision

        :param keyword: Search keyword
        :param max_pages: Maximum number of pages to scrape
        """
        # Create tables if they don't exist
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        current_page = 1

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Prepare payload
                    payload = self._create_payload(
                        keyword=keyword,
                        page=current_page
                    )

                    # Make request
                    async with session.post(
                            self.BASE_URL,
                            json=payload,
                            timeout=10
                    ) as response:
                        response.raise_for_status()
                        json_response = await response.json()
                        job_posts = json_response["data"]["jobPosts"]

                    # Break if no jobs found
                    if not job_posts:
                        logger.info("No more jobs found. Stopping scraping.")
                        break

                    # Process job posts concurrently
                    await self._process_job_posts(session, job_posts)

                    # Move to next page
                    current_page += 1

                except Exception as e:
                    logger.error(f"Error on page {current_page}: {e}")
                    break

    async def export_to_xlsx(self, output_path: str):
        """
        Export job posts from database to XLSX file

        :param output_path: Path to save the XLSX file
        """
        async with self.AsyncSessionLocal() as session:
            # Fetch all job posts
            result = await session.execute(select(JobPost))
            job_posts = result.scalars().all()

            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'Job ID': job.job_vision_id,
                    'Title': job.title,
                    'Company': job.company,
                    'Is Remote': job.is_remote,
                    'Is Internship': job.is_internship,
                    'Salary Range': job.salary_range,
                    'Work Experience': job.work_experience,
                    'Created At': job.created_at
                } for job in job_posts
            ])

            # Export to XLSX
            df.to_excel(output_path, index=False)
            logger.info(f"Exported {len(job_posts)} jobs to {output_path}")


# Typer CLI App
app = typer.Typer(help="JobVision Job Scraper CLI")


@app.command()
def scrape(
    keyword: str = typer.Option("backend", help="Keyword to search for jobs"),
    database: str = typer.Option("job_posts.db", help="Path to SQLite database"),
):
    """Scrape job posts from JobVision"""
    scraper = AsyncJobVisionScraper(database_path=database)
    asyncio.run(scraper.scrape_jobs(keyword=keyword))
    typer.echo(f"Scraping completed for keyword: {keyword}")


@app.command()
def export(
    output: str = typer.Option("job_posts.xlsx", help="Path to export XLSX file"),
    database: str = typer.Option("job_posts.db", help="Path to SQLite database"),
):
    """Export job posts to XLSX file"""
    scraper = AsyncJobVisionScraper(database_path=database)
    asyncio.run(scraper.export_to_xlsx(output_path=output))
    typer.echo(f"Exported job posts to {output}")


def main():
    """Main entry point for the application"""
    app()


if __name__ == "__main__":
    main()