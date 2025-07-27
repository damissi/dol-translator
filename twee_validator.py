import re
import difflib
from pathlib import Path
from collections import defaultdict
import spacy
import ahocorasick
from kiwipiepy import Kiwi
from typing import List, Tuple, Optional, Set

class TweeL10nValidator:
    """
    .twee 파일의 원본과 번역본을 비교하여 로컬라이제이션 품질을 검증하는 종합 클래스.
    NLP와 Aho-Corasick 알고리즘을 사용하여 구조, 구문, 플레이 가능성, 규칙 준수를
    종합적으로 검사하고 상세한 리포트를 생성합니다.
    """

    # --- 상수 정의 ---
    UNTRANSLATED_LINK_WORD_THRESHOLD = 4
    ENGLISH_RATIO_THRESHOLD = 0.8
    CONTEXT_LINES = 2

    # --- 정규표현식 ---
    REGEX = {
        "passage_header": re.compile(r"^(::\s.*)$"),
        "macro": re.compile(r"<<.*?>>"),
        "variable": re.compile(r"\$[a-zA-Z0-9_.]+"),
        "link_with_dest": re.compile(r"\[\[(.*?)\|(.*?)\]\]"),
        "link_simple": re.compile(r"\[\[(.*?)\]\]"),
        "link": re.compile(r"\[\[.*?\]\]"),
        "string_literal": re.compile(r'["\'](.*?)["\']'),
        "korean": re.compile(r"[가-힣]"),
        "english_only": re.compile(r"^[a-zA-Z\s.,!?'\"():<>_`~@#$%^&*=\[\]{}|\\/+-]+$"),
        "word_tokenizer": re.compile(r"[\w']+"),
        "corrupted_char": re.compile(r"�"),
        "forbidden_pattern": re.compile(r"[가-힣]+\s*\([A-Za-z\s]+\)"),
        "code_only_line": re.compile(r"^\s*(<<.*>>|/\*.*?\*/|<!--.*?-->)\s*$"),
        "markdown_header": re.compile(r"^(#+)\s.*$"),
        "html_tag": re.compile(r"<.*?>"),
        "comment": re.compile(r"^(/\*.*?\*/|<!--.*?-->)"),
    }
    REGEX["code_block"] = re.compile(f"({REGEX['macro'].pattern}|{REGEX['link'].pattern}|{REGEX['variable'].pattern}|{REGEX['html_tag'].pattern})")

    # --- 화이트리스트 ---
    ALLOWED_POSTPOSITIONS = frozenset([
        "은", "는", "이", "가", "을", "를", "과", "와", "의", "께", "에게", "한테",
        "으로", "로", "에서", "부터", "까지", "만", "도", "뿐", "이라", "라",
        "이여", "여", "이시여", "시여", "아", "야"
    ])

    def __init__(self, original_path: Path, translated_path: Path, glossary_path: Optional[Path]):
        print("검증기 초기화 중... (NLP 모델 및 용어집 로딩)")
        self.original_path = original_path
        self.translated_path = translated_path
        self.glossary_path = glossary_path
        self.issues = []

        self.nlp_en = spacy.load("en_core_web_sm")
        self.kiwi = Kiwi()
        self.glossary_automaton = ahocorasick.Automaton()

        self._load_files()
        self._build_glossary_automaton()
        print("초기화 완료.")

    def _load_files(self):
        try:
            self.original_lines = self.original_path.read_text('utf-8').splitlines()
            self.translated_lines = self.translated_path.read_text('utf-8').splitlines()
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다 - {e}")
            exit(1)

    def _build_glossary_automaton(self):
        self.glossary = {}
        if self.glossary_path and self.glossary_path.exists():
            lines = self.glossary_path.read_text('utf-8').splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith('#') or not line.strip() or ':' not in line:
                    continue
                parts = [p.strip() for p in line.split(':', 1)]
                if len(parts) == 2 and parts[0] and parts[1]:
                    eng_key, kor_value = parts[0], parts[1]
                    keys_to_add = {eng_key, eng_key.lower(), eng_key.capitalize()}
                    for key in keys_to_add:
                        if key not in self.glossary:
                            self.glossary[key] = kor_value
                            self.glossary_automaton.add_word(key, (key, kor_value))
        
        self.glossary_automaton.make_automaton()
        if not self.glossary and self.glossary_path:
             print(f"경고: 용어집 파일을 찾지 못했거나 내용이 비어있습니다: '{self.glossary_path}'")

    def _add_issue(self, **kwargs):
        self.issues.append(kwargs)

    def _get_pure_text(self, line: str) -> str:
        return self.REGEX["code_block"].sub("", line)

    def _classify_line(self, line: str) -> str:
        """라인의 유형을 분류하여 반환합니다."""
        if not line.strip(): return "BLANK"
        if self.REGEX["passage_header"].match(line): return "PASSAGE_HEADER"
        if self.REGEX["markdown_header"].match(line): return "MARKDOWN_HEADER"
        if self.REGEX["comment"].match(line.strip()): return "COMMENT"
        if not self._get_pure_text(line).strip(): return "PURE_CODE"
        if not self.REGEX["code_block"].search(line): return "PURE_TEXT"
        return "MIXED_CONTENT"

    def run_all_checks(self):
        """모든 검증 단계를 순서대로 실행합니다."""
        print("\n--- 1단계: 구조적 무결성 검사 시작 ---")
        is_structurally_sound = self._check_line_count_and_structure()
        self._check_core_identifiers()
        
        print("\n--- 2단계: 구문, 플레이 가능성, 규칙 준수 검사 시작 ---")
        self._check_global_variable_consistency()
        self._check_all_lines(is_structurally_sound)
        
        print(f"\n모든 검증 완료. 총 {len(self.issues)}개의 문제 발견.")

    def _check_line_count_and_structure(self) -> bool:
        if len(self.original_lines) == len(self.translated_lines):
            return True
        self._add_issue(
            line_num=0, severity="CRITICAL", type="구조적 오류",
            description=f"파일의 전체 줄 수가 일치하지 않습니다. (원본: {len(self.original_lines)}줄, 번역본: {len(self.translated_lines)}줄)"
        )
        matcher = difflib.SequenceMatcher(None, self.original_lines, self.translated_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' or tag == 'replace': continue
            diff_lines = []
            start = max(0, i1 - self.CONTEXT_LINES)
            for i in range(start, i1): diff_lines.append(f"  {self.original_lines[i]}")
            description = ""
            if tag == 'delete':
                description = f"원본 {i1+1}번째 줄 근처의 내용이 번역본에서 삭제되었습니다."
                for i in range(i1, i2): diff_lines.append(f"- {self.original_lines[i]}")
            elif tag == 'insert':
                description = f"번역본 {j1+1}번째 줄 근처에 원본에 없는 내용이 추가되었습니다."
                for j in range(j1, j2): diff_lines.append(f"+ {self.translated_lines[j]}")
            end = min(len(self.original_lines), i2 + self.CONTEXT_LINES)
            for i in range(i2, end): diff_lines.append(f"  {self.original_lines[i]}")
            self._add_issue(
                line_num=i1 + 1, severity="CRITICAL", type="구조적 불일치 (줄 삭제/추가)",
                description=description, diff_text="\n".join(diff_lines)
            )
        return False

    def _check_core_identifiers(self):
        orig_headers = [h for h, l in self._extract_identifiers(self.original_lines, "passage_header")]
        trans_headers = [h for h, l in self._extract_identifiers(self.translated_lines, "passage_header")]
        if orig_headers != trans_headers:
             self._add_issue(line_num=0, severity="CRITICAL", type="패시지 헤더 불일치",
                             description="패시지 헤더의 순서나 내용이 원본과 다릅니다. 게임 링크가 깨질 수 있습니다.")
        orig_dests = {d for d, l in self._extract_identifiers(self.original_lines, "link_destination")}
        trans_dests = {d for d, l in self._extract_identifiers(self.translated_lines, "link_destination")}
        if orig_dests != trans_dests:
            missing = orig_dests - trans_dests
            added = trans_dests - orig_dests
            desc = "링크 목적지 목록이 일치하지 않습니다."
            if missing: desc += f" 누락: {', '.join(sorted(list(missing))[:5])} 등"
            if added: desc += f" 추가/오타: {', '.join(sorted(list(added))[:5])} 등"
            self._add_issue(line_num=0, severity="CRITICAL", type="링크 목적지 불일치", description=desc)

    def _extract_identifiers(self, lines: List[str], id_type: str) -> List[Tuple[str, int]]:
        extracted = []
        for i, line in enumerate(lines):
            if id_type == "passage_header":
                if match := self.REGEX["passage_header"].match(line):
                    extracted.append((match.group(1).strip(), i + 1))
            elif id_type == "link_destination":
                for _, dest in self.REGEX["link_with_dest"].findall(line):
                    extracted.append((dest.strip(), i + 1))
                for dest in self.REGEX["link_simple"].findall(line):
                    if '|' not in dest: extracted.append((dest.strip(), i + 1))
        return extracted

    def _check_global_variable_consistency(self):
        original_vars = set(self.REGEX["variable"].findall("\n".join(self.original_lines)))
        translated_vars = set(self.REGEX["variable"].findall("\n".join(self.translated_lines)))
        if original_vars != translated_vars:
            missing = original_vars - translated_vars
            added = translated_vars - original_vars
            desc = "전체 변수 목록이 일치하지 않습니다."
            if missing: desc += f" 누락된 변수: {', '.join(sorted(list(missing))[:5])} 등"
            if added: desc += f" 추가/손상된 변수: {', '.join(sorted(list(added))[:5])} 등"
            self._add_issue(line_num=0, severity="CRITICAL", type="전역 변수 불일치", description=desc)

    def _check_all_lines(self, is_structurally_sound: bool):
        """모든 라인을 순회하며 유형에 맞는 검사를 수행합니다."""
        for i, translated_line in enumerate(self.translated_lines):
            line_num = i + 1
            original_line = self.original_lines[i] if is_structurally_sound else ""
            
            line_type = self._classify_line(translated_line)
            orig_line_type = self._classify_line(original_line) if is_structurally_sound else None

            # 1. 번역되면 안 되는 라인 유형 검사
            if line_type in ["PASSAGE_HEADER", "MARKDOWN_HEADER", "COMMENT", "BLANK"]:
                if is_structurally_sound and original_line != translated_line:
                    self._add_issue(severity="CRITICAL", type="코드 라인 불일치",
                                    description=f"'{line_type}' 유형의 라인은 원본과 동일해야 합니다.",
                                    line_num=line_num, original=original_line, translated=translated_line)
                continue

            # 2. 콘텐츠가 포함된 라인 검사
            if line_type in ["PURE_TEXT", "MIXED_CONTENT"]:
                self._check_links_for_playability(translated_line, line_num)
                self._check_untranslated_content(original_line, translated_line, line_num, is_structurally_sound)
                self._check_forbidden_patterns(translated_line, line_num)
                if is_structurally_sound:
                    self._check_glossary_compliance_nlp(original_line, translated_line, line_num)

            # 3. 모든 라인 대상 검사
            self._check_text_corruption(translated_line, line_num)
            
            # 4. 코드 라인 무결성 검사 (구조가 같을 때만)
            if is_structurally_sound:
                self._check_macro_corruption(original_line, translated_line, line_num)

    def _check_macro_corruption(self, original_line, translated_line, line_num):
        """매크로 내부 문자열 리터럴 번역을 검사합니다."""
        original_macros = self.REGEX["macro"].findall(original_line)
        translated_macros = self.REGEX["macro"].findall(translated_line)
        if len(original_macros) == len(translated_macros):
            for orig_macro, trans_macro in zip(original_macros, translated_macros):
                # 동적 표현식(예: "text" + var) 내부의 문자열도 검사
                content = trans_macro[2:-2].strip()
                if '+' in content:
                    literals = self.REGEX["string_literal"].findall(content)
                else: # 단순 문자열
                    literals = self.REGEX["string_literal"].findall(trans_macro)

                for literal in literals:
                    if self.REGEX["korean"].search(literal) and literal not in self.ALLOWED_POSTPOSITIONS:
                        self._add_issue(
                            severity="CRITICAL", type="매크로 코드 손상",
                            description=f"매크로 내부 코드 식별자(문자열) '{literal}'이(가) 번역되었습니다.",
                            line_num=line_num, original=f"`{orig_macro}`", translated=f"`{trans_macro}`"
                        )
                        break

    def _check_links_for_playability(self, line, line_num):
        """링크의 표시 텍스트가 비어있거나 번역되지 않았는지 검사합니다."""
        all_links = self.REGEX["link_with_dest"].findall(line) + [(m, m) for m in self.REGEX["link_simple"].findall(line) if '|' not in m]
        for display_text, dest in all_links:
            if not display_text.strip():
                self._add_issue(severity="WARNING", type="빈 상호작용",
                                description="플레이어가 클릭할 수 없는 '빈 링크'가 발견되었습니다.",
                                line_num=line_num, translated=line)
            elif self.REGEX["english_only"].match(self._get_pure_text(display_text)):
                word_count = len(self.REGEX["word_tokenizer"].findall(display_text))
                severity = "WARNING" if word_count >= self.UNTRANSLATED_LINK_WORD_THRESHOLD else "INFO"
                self._add_issue(severity=severity, type="미번역 의심 (링크)",
                                description=f"링크 표시 텍스트 '{display_text}'이(가) 번역되지 않은 것 같습니다.",
                                line_num=line_num, translated=line)

    def _check_untranslated_content(self, original_line, translated_line, line_num, is_structurally_sound):
        """순수 텍스트 라인이 번역되지 않았는지 비율 기반으로 검사합니다."""
        pure_translated = self._get_pure_text(translated_line)
        if not pure_translated.strip() or self.REGEX["korean"].search(pure_translated):
            return
        
        words = self.REGEX["word_tokenizer"].findall(pure_translated)
        if not words: return

        english_words = sum(1 for word in words if self.REGEX["english_only"].match(word) and not word.isdigit())
        if (english_words / len(words)) >= self.ENGLISH_RATIO_THRESHOLD:
            severity = "WARNING" if is_structurally_sound and original_line.strip() == translated_line.strip() else "INFO"
            desc = "이 라인은 번역이 누락되었거나(원본과 동일), 대부분이 영어로 구성되어 검토가 필요합니다."
            self._add_issue(
                severity=severity, type="미번역 의심 (콘텐츠)", description=desc,
                line_num=line_num, original=original_line, translated=translated_line
            )

    def _check_text_corruption(self, line, line_num):
        """텍스트 깨짐 현상을 탐지합니다."""
        if self.REGEX["corrupted_char"].search(line):
            self._add_issue(
                severity="CRITICAL", type="텍스트 손상",
                description="파일 인코딩 문제로 인해 깨진 문자(�)가 발견되었습니다.",
                line_num=line_num, translated=line
            )

    def _check_forbidden_patterns(self, line, line_num):
        """금지된 번역 패턴(원문 병기)을 탐지합니다."""
        if self.REGEX["forbidden_pattern"].search(line):
            self._add_issue(
                severity="WARNING", type="금지된 패턴 사용",
                description="번역문 뒤에 괄호를 사용한 원문 병기 패턴이 발견되었습니다.",
                line_num=line_num, translated=line
            )

    def _check_glossary_compliance_nlp(self, original_line, translated_line, line_num):
        """NLP와 Aho-Corasick을 사용하여 용어집 준수 여부를 정밀 검사합니다."""
        if not self.glossary: return

        pure_original = self._get_pure_text(original_line)
        if not pure_original.strip(): return

        found_eng_terms = {item[1][0] for item in self.glossary_automaton.iter(pure_original)}
        if not found_eng_terms: return

        pure_translated = self._get_pure_text(translated_line)
        tokens = self.kiwi.tokenize(pure_translated)
        found_kor_tokens = {token.form for token in tokens}

        for eng_key in found_eng_terms:
            kor_value = self.glossary.get(eng_key)
            if not kor_value: continue

            if kor_value not in found_kor_tokens and kor_value not in pure_translated:
                if eng_key.lower() in pure_translated.lower():
                    self._add_issue(
                        severity="INFO", type="용어집 미적용",
                        description=f"용어집 단어 '{eng_key}'가 번역되지 않고 원문에 남아있습니다.",
                        line_num=line_num, original=original_line, translated=translated_line
                    )
                else:
                    self._add_issue(
                        severity="WARNING", type="용어집 오역/누락 의심",
                        description=f"용어집 단어 '{eng_key}'의 번역 '{kor_value}'이(가) 누락되었거나 다른 단어로 번역된 것 같습니다.",
                        line_num=line_num, original=original_line, translated=translated_line
                    )

    def generate_report(self, output_path: Path):
        """검증 결과를 Markdown 파일로 생성합니다."""
        if not self.issues:
            report_content = f"# ✅ 검증 완료: {self.translated_path.name}\n\n**축하합니다! 발견된 문제가 없습니다.**"
        else:
            severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            sorted_issues = sorted(self.issues, key=lambda x: (severity_order.get(x["severity"], 99), x["line_num"]))
            
            summary = defaultdict(int)
            for issue in self.issues: summary[issue["severity"]] += 1

            report_content = f"# ❗ 종합 검증 리포트: {self.translated_path.name}\n\n"
            report_content += f"## 요약\n\n- **총 문제 수: {len(self.issues)}**\n"
            if summary["CRITICAL"] > 0: report_content += f"- 🔴 **치명적 오류 (CRITICAL): {summary['CRITICAL']}**\n"
            if summary["WARNING"] > 0: report_content += f"- 🟡 **경고 (WARNING): {summary['WARNING']}**\n"
            if summary["INFO"] > 0: report_content += f"- 🔵 **정보 (INFO): {summary['INFO']}**\n"
            
            report_content += "\n---\n\n## 상세 내용\n\n"

            for issue in sorted_issues:
                icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(issue["severity"], "⚪️")
                line_info = f"(원본 기준 Line: {issue['line_num']})" if issue['line_num'] > 0 else "(전역 검사)"
                report_content += f"### {icon} [{issue['severity']}] {issue['type']} {line_info}\n\n"
                report_content += f"- **문제 설명:** {issue['description']}\n"
                if issue.get('diff_text'):
                    report_content += f"\n**차이점 분석 (Diff):**\n```diff\n{issue['diff_text']}\n```\n"
                if issue.get('original'):
                    report_content += f"- **원본:** `{issue.get('original')}`\n"
                if issue.get('translated'):
                    report_content += f"- **번역본:** `{issue.get('translated')}`\n"
                report_content += "\n---\n"
        
        output_path.write_text(report_content, 'utf-8')
        print(f"\n리포트가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # --- 설정: 여기에 검증할 파일 경로를 직접 입력하세요. ---
    ORIGINAL_FILE_PATH =
    TRANSLATED_FILE_PATH = 
    OUTPUT_REPORT_PATH = 
    GLOSSARY_FILE_PATH =
    # ----------------------------------------------------

    original_p = Path(ORIGINAL_FILE_PATH)
    translated_p = Path(TRANSLATED_FILE_PATH)
    glossary_p = Path(GLOSSARY_FILE_PATH) if GLOSSARY_FILE_PATH and Path(GLOSSARY_FILE_PATH).exists() else None
    output_p = Path(OUTPUT_REPORT_PATH)

    validator = TweeL10nValidator(original_path=original_p, translated_path=translated_p, glossary_path=glossary_p)
    validator.run_all_checks()
    validator.generate_report(output_path=output_p)



