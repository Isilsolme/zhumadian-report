#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
爬取驻马店市政府工作报告
从 https://www.zhumadian.gov.cn/zwgk/zfxxgk/fdzdgknr/qtfdxx/zfgzbg/ 抓取历年报告文本
"""

import os
import re
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup


class ZhuMaDianReportScraper:
    """驻马店市政府工作报告爬虫"""

    def __init__(self):
        self.base_url = "https://www.zhumadian.gov.cn"
        self.archive_url = "https://www.zhumadian.gov.cn/zwgk/zfxxgk/fdzdgknr/qtfdxx/zfgzbg/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }
        self.output_dir = "zhumadian_reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_html(self, url):
        """下载页面HTML"""
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            text = resp.text
            # 处理可能的编码问题
            if "\ufffd" in text or "Ã" in text:
                resp.encoding = resp.apparent_encoding
                text = resp.text
            return text
        except requests.RequestException as e:
            print(f"请求失败 {url}: {e}")
            return None

    def _archive_page_urls(self, max_pages=30):
        """生成列表页URL"""
        urls = [self.archive_url]
        for i in range(1, max_pages):
            urls.append(urljoin(self.archive_url, f"index_{i}.html"))
        return urls

    def parse_archive_page(self, html):
        """解析列表页，提取报告标题、链接和年份"""
        soup = BeautifulSoup(html, "html.parser")
        reports = []
        # 多种可能的容器
        containers = [
            soup.find("div", class_="zfxxgk_zdgkc"),
            soup.find("ul", class_="zfxxgk_list"),
            soup.find("div", class_="list"),
            soup,
        ]
        links = []
        for container in containers:
            if container:
                links = container.find_all("a", href=True)
                if links:
                    break

        for a_tag in links:
            title = a_tag.get_text(strip=True)
            href = a_tag.get("href", "").strip()
            if not href or "政府工作报告" not in title:
                continue
            url = urljoin(self.archive_url, href)
            # 提取年份
            year_match = re.search(r"(20\d{2})", title)
            if not year_match:
                year_match = re.search(r"/t(20\d{2})", url)
            if not year_match:
                year_match = re.search(r"(20\d{2})", url)
            if not year_match:
                continue
            reports.append({"title": title, "url": url, "year": year_match.group(1)})

        # 去重
        seen = set()
        unique = []
        for r in reports:
            if r["url"] not in seen:
                seen.add(r["url"])
                unique.append(r)
        return unique

    def _is_heading_line(self, line):
        """判断是否为小标题行"""
        line = line.strip()
        if not line:
            return False
        patterns = [
            r"^[一二三四五六七八九十]+、",
            r"^（[一二三四五六七八九十]+）",
            r"^\([一二三四五六七八九十]+\)",
            r"^第[一二三四五六七八九十百]+[章节部分]",
            r"^各位代表",
            r"^政\s*府\s*工\s*作\s*报\s*告",
            r"^——\d{4}年",
        ]
        return any(re.search(p, line) for p in patterns)

    def _normalize_line(self, line):
        """规范化单行文本"""
        line = line.replace("\xa0", " ").strip()
        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", line)
        line = re.sub(r"(?<=\d)\s+(?=\d)", "", line)
        line = re.sub(r"(?<=\d)\s+(?=[亿万千百十%元])", "", line)
        line = re.sub(r"(?<=[亿万千百十%元])\s+(?=\d)", "", line)
        return line

    def _clean_block_text(self, text):
        """将正文块整理为段落文本"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = [self._normalize_line(line) for line in text.split("\n")]
        paragraphs = []
        current = ""
        for line in raw_lines:
            if not line:
                continue
            if not current:
                current = line
                continue
            if self._is_heading_line(line):
                paragraphs.append(current)
                current = line
                continue
            current += line
        if current:
            paragraphs.append(current)
        cleaned = [p.strip() for p in paragraphs if len(p.strip()) >= 2]
        return "\n\n".join(cleaned)

    def _is_good_report_text(self, text):
        """简单校验是否为有效报告正文"""
        if not text or len(text) < 500:
            return False
        keywords = ["政府工作报告", "各位代表", "工作回顾", "主要任务"]
        return sum(1 for k in keywords if k in text) >= 2

    def extract_report_content(self, html):
        """从详情页HTML中提取报告正文"""
        soup = BeautifulSoup(html, "html.parser")
        content_div = (
            soup.select_one(".lis_list_part2_content .trs_import_qwhych")
            or soup.select_one(".lis_list_part2_content")
        )
        if not content_div:
            for selector in ["div.article", "div.content", "div.TRS_Editor", "article", "main"]:
                content_div = soup.select_one(selector)
                if content_div:
                    break
        if content_div:
            for node in content_div(["script", "style", "nav", "header", "footer", "aside"]):
                node.decompose()
            paragraphs = content_div.find_all("p")
            if paragraphs:
                clean_ps = [p.get_text(separator="\n", strip=True) for p in paragraphs if p.get_text(strip=True)]
                if clean_ps:
                    joined = self._clean_block_text("\n".join(clean_ps))
                    if self._is_good_report_text(joined):
                        return joined
            raw = content_div.get_text(separator="\n", strip=True)
            cleaned = self._clean_block_text(raw)
            if self._is_good_report_text(cleaned):
                return cleaned

        # 回退：取最长文本块
        best = ""
        for tag in soup.find_all(["div", "article", "section", "main"]):
            for node in tag(["script", "style", "nav", "header", "footer", "aside"]):
                node.decompose()
            candidate = self._clean_block_text(tag.get_text(separator="\n", strip=True))
            if len(candidate) > len(best):
                best = candidate
        if self._is_good_report_text(best):
            return best
        return best if len(best) >= 1000 else None

    def save_report_to_file(self, year, title, content):
        """保存报告到本地文件"""
        safe_title = re.sub(r"[\\/*?:\"<>|]", "", title)
        filename = os.path.join(self.output_dir, f"{year}_{safe_title}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"标题：{title}\n年份：{year}\n{'=' * 60}\n\n{content}")
        print(f"  已保存: {filename}")

    def crawl_reports(self, max_reports=50, max_pages=30, sleep_seconds=1.0):
        """主流程：抓取并保存报告"""
        print("=" * 60)
        print("开始爬取驻马店市历年政府工作报告")
        print("=" * 60)

        all_reports = []
        for idx, page_url in enumerate(self._archive_page_urls(max_pages), 1):
            print(f"[{idx}/{max_pages}] 获取列表页: {page_url}")
            html = self.fetch_html(page_url)
            if not html:
                continue
            page_reports = self.parse_archive_page(html)
            if page_reports:
                print(f"  解析到 {len(page_reports)} 条候选报告")
                all_reports.extend(page_reports)

        # 去重并按年份排序
        dedup = {r["url"]: r for r in all_reports}
        reports = list(dedup.values())
        if not reports:
            print("未找到任何报告链接")
            return []
        reports.sort(key=lambda x: int(x["year"]), reverse=True)
        reports = reports[:max_reports]
        print(f"共得到 {len(reports)} 个报告链接，开始抓取正文")

        results = []
        for i, report in enumerate(reports, 1):
            print(f"\n[{i}/{len(reports)}] {report['year']}年 {report['title']}")
            detail_html = self.fetch_html(report["url"])
            if not detail_html:
                print("  获取详情页失败")
                continue
            content = self.extract_report_content(detail_html)
            if not content:
                print("  正文提取失败")
                continue
            self.save_report_to_file(report["year"], report["title"], content)
            print(f"  成功，正文长度: {len(content)}")
            results.append(report)
            time.sleep(sleep_seconds)

        print(f"\n完成：成功保存 {len(results)} 份报告")
        return results


def main():
    scraper = ZhuMaDianReportScraper()
    scraper.crawl_reports(max_reports=25, max_pages=3, sleep_seconds=1.0)


if __name__ == "__main__":
    main()