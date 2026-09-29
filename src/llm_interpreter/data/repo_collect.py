import base64
from pathlib import Path
from typing import List, Tuple, Union

import spdx_lookup as lookup
import seutil as su
from github import UnknownObjectException
from jsonargparse.typing import Path_dc, Path_drw
from seutil.project import Project
from tqdm import tqdm

from llm_interpreter.macros import Macros

logger = su.log.get_logger(__name__, su.log.INFO)


class ReposCollector:
    """
    Collection and filtering of repositories.
    """

    TAG_TIME_LIMIT = 1577836800  # 2020-01-01 00:00 UTC
    CLONE_TIMEOUT = 300

    def __init__(
        self, downloads_dir: Union[Path_drw, Path_dc, Path] = Macros.downloads_dir
    ):
        if not isinstance(downloads_dir, Path):
            downloads_dir = Path(downloads_dir.abs_path)
        self.downloads_dir = downloads_dir

    def download_repo(self, lists: List[str], out_dir: Path):
        """
        Download the repository from the given link.
        """
        repo2source = {}
        for lst in lists:
            for repo_name in su.io.load(lst, su.io.Fmt.txtList):
                if "github.com" in repo_name:
                    repo_name = repo_name.split("github.com/")[-1].replace(".git", "")
                if repo_name in repo2source:
                    logger.warning(
                        f"{repo_name} already in {repo2source[repo_name]}, appear again in {lst.stem}"
                    )
                repo2source[repo_name] = lst.stem
        print(f"In total {len(repo2source)} repos to search")

        success, failed = 0, 0
        projects = []
        logs = {"success_projects": [], "failed_projects": []}
        for repo_name, source in tqdm(repo2source.items(), desc="Downloading repos"):
            try:
                p = self.download_repo_from_github(source, repo_name)
                projects.append(p)
                logs["success_projects"].append(p.full_name)
                success += 1
            except Exception as e:
                logger.error(f"Error downloading {repo_name} from {source}: {e}")
                logs["failed_projects"].append(repo_name)
                failed += 1

        print(f"Successfully downloaded {success} repos")
        print(f"Failed to download {failed} repos")
        su.io.dump(out_dir / "repos.json", projects, su.io.Fmt.jsonPretty)
        su.io.dump(out_dir / "logs.json", logs, su.io.Fmt.jsonPretty)

    def download_repo_from_github(self, sources: List, full_name_raw: str) -> Project:
        """
        Download the repository from the given link.
        """
        repo = su.GitHubUtils.ensure_github_api_call(
            lambda g: g.get_repo(full_name_raw), max_retry_times=2
        )
        url = su.GitHubUtils.ensure_github_api_call(lambda g: repo.clone_url)
        # get the latest user/repo name from GitHub API
        user_name = su.GitHubUtils.ensure_github_api_call(
            lambda g: repo.owner.login, max_retry_times=2
        )
        repo_name = su.GitHubUtils.ensure_github_api_call(
            lambda g: repo.name, max_retry_times=2
        )
        full_name = f"{user_name}_{repo_name}"
        p = Project(full_name=full_name, url=url)
        p.data["repo"] = repo_name
        p.data["user"] = user_name
        p.data["branch"] = su.GitHubUtils.ensure_github_api_call(
            lambda g: repo.default_branch, max_retry_times=2
        )
        p.data["stars"] = su.GitHubUtils.ensure_github_api_call(
            lambda g: repo.stargazers_count, max_retry_times=2
        )
        p.data["sources"] = sources
        # license
        license_type, license_text = self.get_project_license(p)
        p.data["license_type"] = license_type
        p.data["license_text"] = license_text

        # Try to download the project; remove previously downloaded repo
        with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
            p.clone(self.downloads_dir)

        # Select a sha: either a tag (approximation of release) after 2020.1.1, or the latest commit
        p.checkout(p.data["branch"], forced=True)
        with su.io.cd(p.dir):
            rr = su.bash.run("git describe --tags $(git rev-list --tags --max-count=1)")
            if rr.returncode == 0:
                tag = rr.stdout.strip()
                tag_time = int(
                    su.bash.run(f"git log {tag} -1 --pretty='%at'", 0).stdout.strip()
                )
                if tag_time >= self.TAG_TIME_LIMIT:
                    p.checkout(tag, forced=True)
                    p.data["tag"] = tag
                    p.data["tag_time"] = tag_time

        p.data["sha"] = p.get_cur_revision()
        return p

    allowed_licenses = [
        "Apache License 2.0",
        'BSD 2-clause "Simplified" License',
        'BSD 3-clause "New" or "Revised" License',
        "Common Public License 1.0",
        "Creative Commons Attribution Share Alike 4.0",
        "Creative Commons Zero v1.0 Universal",
        "GNU Affero General Public License v3.0",
        "GNU General Public License v2.0 only",
        "GNU General Public License v3.0 only",
        "GNU Lesser General Public License v2.1 only",
        "GNU Lesser General Public License v3.0 only",
        "ISC License",
        "MIT License",
        "Mozilla Public License 2.0",
    ]

    @classmethod
    def get_project_license(cls, p: Project) -> Tuple[str, str]:
        """
        Gets the license of the project.
        :param p: the Project instance.
        :return: the project's license type and full text.
        """

        conn = su.GitHubUtils.get_github()

        slug = p.data["user"] + "/" + p.data["repo"]
        try:
            repo = conn.get_repo(slug)
            license_text = base64.b64decode(
                repo.get_license().content.encode()
            ).decode()
            license_info = lookup.match(license_text)
            if license_info is not None:
                license_type = str(license_info.license)
            else:
                license_type = "Unknown"
        except UnknownObjectException:
            license_type = "Unknown"
            license_text = None

        return license_type, license_text
