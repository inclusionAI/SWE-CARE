"""
BM25-based retrieval utilities for code search and file content extraction.

This module provides functionality for retrieving relevant code files using BM25 search,
managing Git repositories at specific commits, and extracting file contents for analysis.

Inspired and modified from:
https://github.com/SWE-bench/SWE-bench/blob/main/swebench/inference/make_datasets/bm25_retrieval.py

"""

import ast
import json
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import jedi
from filelock import FileLock
from git import Repo
from loguru import logger
from tqdm.auto import tqdm

from swe_care.utils.read_file import read_file_to_string


class ContextManager:
    """
    A context manager for managing a Git repository at a specific commit using git worktree.

    This implementation creates a temporary worktree for each usage to avoid conflicts
    when multiple threads/processes need to access different commits of the same repository.
    The worktree is automatically cleaned up when the context manager exits.

    Args:
        repo_path (str): The path to the Git repository.
        base_commit (str): The commit hash to switch to.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Attributes:
        repo_path (str): The path to the Git repository.
        base_commit (str): The commit hash to switch to.
        verbose (bool): Whether to print verbose output.
        repo (git.Repo): The Git repository object.
        worktree_path (str): Path to the temporary worktree.
        original_repo_path (str): Original repository path before worktree creation.
        temp_dir (str): Path to the temporary directory containing the worktree.

    Methods:
        __enter__(): Creates a temporary worktree at the specified commit and returns the context manager object.
        get_readme_files(): Returns a list of filenames for all README files in the repository.
        __exit__(exc_type, exc_val, exc_tb): Removes the temporary worktree.
    """

    def __init__(self, repo_path, base_commit, verbose=False):
        self.original_repo_path = Path(repo_path).resolve().as_posix()
        self.base_commit = base_commit
        self.verbose = verbose
        self.repo = Repo(self.original_repo_path)
        self.worktree_path = None
        self.temp_dir = None

    def __enter__(self):
        # Extract repository name from the path
        repo_basename = os.path.basename(self.original_repo_path)

        # Create a unique temporary directory for this worktree
        self.temp_dir = tempfile.mkdtemp(prefix="swe_care_git_worktree_")

        # Create worktree name
        worktree_name = f"{repo_basename}_{self.base_commit[:8]}"
        self.worktree_path = os.path.join(self.temp_dir, worktree_name)

        if self.verbose:
            logger.debug(
                f"Creating temporary worktree at {self.worktree_path} for commit {self.base_commit}"
            )

        try:
            # Create a new worktree at the specified commit
            self.repo.git.worktree("add", "-f", self.worktree_path, self.base_commit)

            # Update repo_path to point to the worktree
            self.repo_path = self.worktree_path

            # Create a new Repo object for the worktree
            self.repo = Repo(self.worktree_path)

        except Exception as e:
            logger.error(f"Failed to create worktree for {self.base_commit}: {str(e)}")
            # Clean up on error
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise e

        return self

    def get_readme_files(self):
        files = os.listdir(self.worktree_path)
        files = list(
            filter(lambda x: os.path.isfile(os.path.join(self.worktree_path, x)), files)
        )
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up the temporary worktree
        if self.worktree_path:
            try:
                # First, use git to remove the worktree
                original_repo = Repo(self.original_repo_path)
                original_repo.git.worktree("remove", "-f", self.worktree_path)
            except Exception as e:
                logger.warning(f"Failed to remove worktree via git: {e}")

        # Clean up the temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {e}")


def contents_only(filename, relative_path):
    """Returns the contents of a file."""
    return read_file_to_string(filename)


def file_name_and_contents(filename, relative_path):
    """Returns the contents of a file along with its relative path."""
    text = relative_path + "\n"
    text += read_file_to_string(filename)
    return text


def file_name_and_documentation(filename, relative_path):
    """Returns the structural documentation of a Python file along with its relative path."""
    text = relative_path + "\n"
    try:
        content = read_file_to_string(filename)
        node = ast.parse(content)
        data = ast.get_docstring(node)
        if data:
            text += f"{data}"
        for child_node in ast.walk(node):
            if isinstance(
                child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                data = ast.get_docstring(child_node)
                if data:
                    text += f"\n\n{child_node.name}\n{data}"
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to parse file {str(filename)}. Using simple filecontent.")
        text += read_file_to_string(filename)
    return text


def file_name_and_docs_jedi(filename, relative_path):
    """Returns the documentation of a Python file using Jedi along with its relative path."""
    text = relative_path + "\n"
    source_code = read_file_to_string(filename)
    try:
        script = jedi.Script(source_code, path=filename)
        module = script.get_context()
        docstring = module.docstring()
        text += f"{module.full_name}\n"
        if docstring:
            text += f"{docstring}\n\n"
        # abspath = Path(filename).absolute()
        names = [
            name
            for name in script.get_names(
                all_scopes=True, definitions=True, references=False
            )
            if not name.in_builtin_module()
        ]
        for name in names:
            try:
                origin = name.goto(follow_imports=True)[0]
                if origin.module_name != module.full_name:
                    continue
                if name.parent().full_name != module.full_name:
                    if name.type in {"statement", "param"}:
                        continue
                full_name = name.full_name
                text += f"{full_name}\n"
                docstring = name.docstring()
                if docstring:
                    text += f"{docstring}\n\n"
            except Exception:
                continue
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed to parse file {str(filename)}. Using simple filecontent.")
        text = f"{relative_path}\n{source_code}"
        return text
    return text


DOCUMENT_ENCODING_FUNCTIONS = {
    "contents_only": contents_only,
    "file_name_and_contents": file_name_and_contents,
    "file_name_and_documentation": file_name_and_documentation,
    "file_name_and_docs_jedi": file_name_and_docs_jedi,
}


def is_test(name, test_phrases=None):
    """Checks if a given filename or path indicates a test file."""
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", name.lower()))
    return any(word in words for word in test_phrases)


def list_files(root_dir, include_tests=False):
    """Lists all Python files in a directory, optionally excluding test files."""
    files = []
    for filename in Path(root_dir).rglob("*.py"):
        # Only check the relative path for test patterns, not the full path
        relative_path = filename.relative_to(root_dir).as_posix()
        if not include_tests and is_test(relative_path):
            continue
        files.append(relative_path)
    return files


def clone_repo(repo, root_dir, token: str | None = None):
    """
    Clones a GitHub repository to a specified directory.

    Args:
        repo (str): The GitHub repository to clone.
        root_dir (str): The root directory to clone the repository to.
        token (str): The GitHub personal access token to use for authentication.

    Returns:
        Path: The path to the cloned repository directory.
    """
    # Create repos subdirectory
    repos_dir = Path(root_dir) / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = repos_dir / f"{repo.replace('/', '__')}"

    # Use file lock to prevent concurrent cloning of the same repository
    clone_lock_path = repos_dir / f"{repo_dir.name}.lock"
    clone_lock = FileLock(str(clone_lock_path))

    with clone_lock:
        if not repo_dir.exists():
            if token:
                repo_url = f"https://{token}@github.com/{repo}.git"
            else:
                repo_url = f"https://github.com/{repo}.git"
            logger.info(f"Cloning {repo} {os.getpid()}")
            Repo.clone_from(repo_url, repo_dir)
    return repo_dir


def build_documents(
    repo_dir, commit, document_encoding_func, include_readmes: bool = False
):
    """
    Builds a dictionary of documents from a given repository directory and commit.

    Args:
        repo_dir (str): The path to the repository directory.
        commit (str): The commit hash to use.
        document_encoding_func (function): A function that takes a filename and a relative path and returns the encoded document text.
        include_readmes (bool): Whether to include README files in the documents.

    Returns:
        dict: A dictionary where the keys are the relative paths of the documents and the values are the encoded document text.
    """
    documents = dict()
    with ContextManager(repo_dir, commit) as ctx:
        filenames = list_files(
            ctx.repo_path, include_tests=False
        )  # Extract all Python files, optionally excluding tests

        if include_readmes:
            readme_files = ctx.get_readme_files()
            # Add readme files to the front of the list
            filenames = readme_files + filenames

        logger.info(
            f"Found {len(filenames)} files in {ctx.repo_path} at commit {commit}"
        )
        for relative_path in filenames:
            filename = os.path.join(ctx.repo_path, relative_path)
            try:
                text = document_encoding_func(filename, relative_path)
                documents[relative_path] = text
            except Exception as e:
                logger.error(f"Failed to encode file {filename}: {e}")
                continue
    return documents


def make_index(
    repo_dir,
    root_dir,
    commit,
    document_encoding_func,
    instance_id,
):
    """
    Builds an index for a given set of documents using Pyserini.

    Args:
        repo_dir (str): The path to the repository directory.
        root_dir (str): The path to the root directory.
        commit (str): The commit hash to use for retrieval.
        document_encoding_func (function): The function to use for encoding documents.
        instance_id (int): The ID of the current instance.

    Returns:
        index_path (Path): The path to the built index.
    """
    # Create indexes subdirectory
    indexes_dir = Path(root_dir) / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    # Extract repo info from instance_id (format: owner__repo_commit)
    index_name = str(instance_id)
    index_path = indexes_dir / index_name / "index"

    # Use file lock to prevent concurrent index creation for the same instance
    index_lock_path = indexes_dir / f"{index_name}.lock"
    index_lock = FileLock(str(index_lock_path))

    with index_lock:
        if index_path.exists():
            return index_path

        # Create documents subdirectory
        documents_dir = Path(root_dir) / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        documents_path = Path(documents_dir, instance_id, "documents.jsonl")

        if not documents_path.parent.exists():
            documents_path.parent.mkdir(parents=True, exist_ok=True)
        documents = build_documents(repo_dir, commit, document_encoding_func)
        with open(documents_path, "w") as docfile:
            for relative_path, contents in documents.items():
                print(
                    json.dumps({"id": relative_path, "contents": contents}),
                    file=docfile,
                    flush=True,
                )

        # Use pyserini's autoclass to ensure proper classpath configuration
        from pyserini.pyclass import autoclass

        # Suppress Java logging output
        JLogManager = autoclass("java.util.logging.LogManager")
        JLevel = autoclass("java.util.logging.Level")

        # Get the root logger and set its level to SEVERE (only show critical errors)
        root_logger = JLogManager.getLogManager().getLogger("")
        root_logger.setLevel(JLevel.SEVERE)

        # Also suppress specific Anserini loggers
        for logger_name in ["io.anserini", "org.apache.lucene"]:
            logger = JLogManager.getLogManager().getLogger(logger_name)
            if logger:
                logger.setLevel(JLevel.SEVERE)

        # Prepare arguments for IndexCollection
        args = [
            "-collection",
            "JsonCollection",
            "-generator",
            "DefaultLuceneDocumentGenerator",
            "-threads",
            "2",
            "-input",
            documents_path.parent.as_posix(),
            "-index",
            index_path.as_posix(),
            "-storePositions",
            "-storeDocvectors",
            "-storeRaw",
            "-quiet",  # Suppress verbose output
        ]

        # Call the Java IndexCollection directly
        JIndexCollection = autoclass("io.anserini.index.IndexCollection")
        JIndexCollection.main(args)
    return index_path


def get_remaining_instances(instances, output_file):
    """
    Filters a list of instances to exclude those that have already been processed and saved in a file.

    Args:
        instances (List[Dict]): A list of instances, where each instance is a dictionary with an "instance_id" key.
        output_file (Path): The path to the file where the processed instances are saved.

    Returns:
        List[Dict]: A list of instances that have not been processed yet.
    """
    instance_ids = set()
    remaining_instances = list()
    if output_file.exists():
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file) as f:
                for line in f:
                    instance = json.loads(line)
                    instance_id = instance["instance_id"]
                    instance_ids.add(instance_id)
            logger.warning(
                f"Found {len(instance_ids)} existing instances in {output_file}. Will skip them."
            )
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return instances
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id not in instance_ids:
            remaining_instances.append(instance)
    return remaining_instances


def search(instance, index_path, k=20):
    """
    Searches for relevant documents in the given index for the given instance.

    Args:
        instance (dict): The instance to search for.
        index_path (str): The path to the index to search in.
        k (int): The number of results to return.

    Returns:
        dict: A dictionary containing the instance ID and a list of hits, where each hit is a dictionary containing the
        document ID and its score.
    """
    from pyserini.search.lucene import LuceneSearcher

    try:
        instance_id = instance["instance_id"]
        searcher = LuceneSearcher(index_path.as_posix())
        cutoff = len(instance["query"])
        while True:
            try:
                hits = searcher.search(
                    instance["query"][:cutoff],
                    k=k,
                    remove_dups=True,
                )
            except Exception as e:
                if "maxClauseCount" in str(e):
                    cutoff = int(round(cutoff * 0.8))
                    continue
                else:
                    raise e
            break
        results = {"instance_id": instance_id, "hits": []}
        for hit in hits:
            doc = json.loads(hit.lucene_document.get("raw"))
            raw = doc.get("contents") or doc.get("text") or doc.get("segment") or ""
            results["hits"].append({"docid": hit.docid, "score": hit.score, "doc": raw})
        return results
    except Exception:
        logger.error(f"Failed to process {instance_id}")
        logger.error(traceback.format_exc())
        return None


def search_indexes(remaining_instance, output_file, all_index_paths):
    """
    Searches the indexes for the given instances and writes the results to the output file.

    Args:
        remaining_instance (list): A list of instances to search for.
        output_file (str): The path to the output file to write the results to.
        all_index_paths (dict): A dictionary mapping instance IDs to the paths of their indexes.
    """
    for instance in tqdm(remaining_instance, desc="Retrieving"):
        instance_id = instance["instance_id"]
        if instance_id not in all_index_paths:
            continue
        index_path = all_index_paths[instance_id]
        results = search(instance, index_path, k=20)
        if results is None:
            continue
        with FileLock(output_file.as_posix() + ".lock"):
            with open(output_file, "a") as out_file:
                print(json.dumps(results), file=out_file, flush=True)


def get_missing_ids(instances, output_file):
    """Checks which instance IDs from the given instances are missing in the output file."""
    with open(output_file) as f:
        written_ids = set()
        for line in f:
            instance = json.loads(line)
            instance_id = instance["instance_id"]
            written_ids.add(instance_id)
    missing_ids = set()
    for instance in instances:
        instance_id = instance["instance_id"]
        if instance_id not in written_ids:
            missing_ids.add(instance_id)
    return missing_ids


def get_index_paths_worker(
    instance,
    root_dir_name,
    document_encoding_func,
    token,
):
    """
    Worker function to process an instance and create an index for it.
    """
    index_path = None
    repo = instance["repo"]
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    try:
        repo_dir = clone_repo(repo, root_dir_name, token)
        index_path = make_index(
            repo_dir=repo_dir,
            root_dir=root_dir_name,
            commit=commit,
            document_encoding_func=document_encoding_func,
            instance_id=instance_id,
        )
    except Exception:
        logger.error(f"Failed to process {repo}/{commit} (instance {instance_id})")
        logger.error(traceback.format_exc())
    return instance_id, index_path


def get_index_paths(
    remaining_instances: list[dict[str, Any]],
    root_dir_name: str,
    document_encoding_func: Any,
    token: str,
    output_file: str,
) -> dict[str, str]:
    """
    Retrieves the index paths for the given instances using multiple processes.

    Args:
        remaining_instances: A list of instances for which to retrieve the index paths.
        root_dir_name: The root directory name.
        document_encoding_func: A function for encoding documents.
        token: The token to use for authentication.
        output_file: The output file.

    Returns:
        A dictionary mapping instance IDs to index paths.
    """
    all_index_paths = dict()
    for instance in tqdm(remaining_instances, desc="Indexing"):
        instance_id, index_path = get_index_paths_worker(
            instance=instance,
            root_dir_name=root_dir_name,
            document_encoding_func=document_encoding_func,
            token=token,
        )
        if index_path is None:
            continue
        all_index_paths[instance_id] = index_path
    return all_index_paths


def get_root_dir(dataset_name, output_dir, document_encoding_style):
    """Creates a root directory for storing indexes based on the dataset name and document encoding style.
    If the directory does not exist, it will be created.
    Args:
        dataset_name (str): The name of the dataset.
        output_dir (str): The base output directory.
        document_encoding_style (str): The style of document encoding to use.
    Returns:
        tuple: A tuple containing the root directory path and the root directory name.
    """
    root_dir = Path(output_dir, dataset_name, document_encoding_style + "_indexes")
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    root_dir_name = root_dir
    return root_dir, root_dir_name
