import ipaddress
import json
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import trafilatura


MAX_REDIRECTS = 5
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 10
USER_AGENT = "NewsSortAI/1.0 academic-prototype"


@dataclass
class ArticleExtractionResult:
    success: bool
    url: str
    final_url: str = ""
    domain: str = ""
    title: str = ""
    author: str = ""
    date: str = ""
    text: str = ""
    word_count: int = 0
    error: str = ""


def extract_article_from_url(url: str) -> ArticleExtractionResult:
    """
    Download a news article page and extract the readable article text.
    """
    normalized_url = _normalize_url(url)
    validation_error = _validate_public_url(normalized_url)
    if validation_error:
        return ArticleExtractionResult(success=False, url=url, error=validation_error)

    try:
        html, final_url = _download_html(normalized_url)
    except requests.Timeout:
        return ArticleExtractionResult(
            success=False,
            url=normalized_url,
            error="The website took too long to respond. Please try again or paste the article text manually.",
        )
    except requests.ConnectionError as error:
        error_message = str(error).lower()
        if "winerror 10013" in error_message or "forbidden by its access permissions" in error_message:
            friendly_error = (
                "This local preview is not allowed to access external websites. "
                "Restart the app from your normal PyCharm terminal, then try the link again."
            )
        else:
            friendly_error = (
                "The article website could not be reached. "
                "Please check the link and internet connection, or paste the article text manually."
            )
        return ArticleExtractionResult(
            success=False,
            url=normalized_url,
            error=friendly_error,
        )
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        if status_code in {401, 403}:
            friendly_error = (
                "This website does not allow automatic article extraction. "
                "Please paste the article text manually."
            )
        elif status_code == 404:
            friendly_error = "The article page could not be found. Please check the link."
        else:
            friendly_error = (
                "The article website returned an error. "
                "Please try again later or paste the article text manually."
            )
        return ArticleExtractionResult(
            success=False,
            url=normalized_url,
            error=friendly_error,
        )
    except requests.RequestException:
        return ArticleExtractionResult(
            success=False,
            url=normalized_url,
            error="The article page could not be downloaded. Please paste the article text manually.",
        )
    except ValueError as error:
        return ArticleExtractionResult(success=False, url=normalized_url, error=str(error))

    metadata_json = trafilatura.extract(
        html,
        url=final_url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_tables=False,
    )

    if not metadata_json:
        return ArticleExtractionResult(
            success=False,
            url=normalized_url,
            final_url=final_url,
            domain=urlparse(final_url).netloc,
            error="The page was opened, but no clear article text could be extracted.",
        )

    article_data = json.loads(metadata_json)
    text = str(article_data.get("text") or "").strip()
    word_count = len(text.split())

    if word_count < 30:
        return ArticleExtractionResult(
            success=False,
            url=normalized_url,
            final_url=final_url,
            domain=urlparse(final_url).netloc,
            text=text,
            word_count=word_count,
            error="The extracted article is too short. Please paste the article text manually.",
        )

    return ArticleExtractionResult(
        success=True,
        url=normalized_url,
        final_url=final_url,
        domain=urlparse(final_url).netloc,
        title=str(article_data.get("title") or "").strip(),
        author=str(article_data.get("author") or "").strip(),
        date=str(article_data.get("date") or "").strip(),
        text=text,
        word_count=word_count,
    )


def _normalize_url(url: str) -> str:
    clean_url = str(url).strip()
    if clean_url and not clean_url.startswith(("http://", "https://")):
        clean_url = f"https://{clean_url}"
    return clean_url


def _validate_public_url(url: str) -> Optional[str]:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        return "Please enter a valid article link that starts with http:// or https://."

    if not parsed_url.hostname:
        return "Please enter a valid article link with a website domain."

    try:
        addresses = socket.getaddrinfo(parsed_url.hostname, None)
    except socket.gaierror:
        return "The website domain could not be found."

    for address in addresses:
        ip_address = ipaddress.ip_address(address[4][0])
        if (
            ip_address.is_private
            or ip_address.is_loopback
            or ip_address.is_link_local
            or ip_address.is_multicast
            or ip_address.is_reserved
            or ip_address.is_unspecified
        ):
            return "For safety, local or private network links are not allowed."

    return None


def _download_html(url: str) -> tuple[str, str]:
    session = requests.Session()
    current_url = url

    for _ in range(MAX_REDIRECTS + 1):
        validation_error = _validate_public_url(current_url)
        if validation_error:
            raise ValueError(validation_error)

        response = session.get(
            current_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=False,
            stream=True,
        )

        if 300 <= response.status_code < 400:
            redirect_target = response.headers.get("Location")
            if not redirect_target:
                raise ValueError("The page redirected, but no new location was provided.")
            current_url = urljoin(current_url, redirect_target)
            continue

        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise ValueError("The link does not look like a normal news article webpage.")

        chunks = []
        total_size = 0
        for chunk in response.iter_content(chunk_size=65536, decode_unicode=False):
            if not chunk:
                continue
            total_size += len(chunk)
            if total_size > MAX_DOWNLOAD_BYTES:
                raise ValueError("The webpage is too large to process in this prototype.")
            chunks.append(chunk)

        response.encoding = response.encoding or "utf-8"
        html = b"".join(chunks).decode(response.encoding, errors="replace")
        return html, response.url

    raise ValueError("The article link redirected too many times.")
