import re
import difflib
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Set

class TweeL10nValidator:
    """
    .twee 파일의 원본과 번역본을 비교하여 로컬라이제이션 품질을 검증하는 종합 클래스.
    1단계(구조)와 2단계(구문) 검사를 포함하며, 확장 가능하도록 설계되었습니다.
    """

    # 미리 컴파일된 정규표현식
    REGEX = {
        "passage_header": re.compile(r"^(::\s.*)$"),
        "macro": re.compile(r"<<.*?>>"),
        "variable": re.compile(r"\$[a-zA-Z0-9_.]+"),
        "link_with_dest": re.compile(r"\[\[(.*?)\|(.*?)\]\]"),
        "link_simple": re.compile(r"\[\[([^|]+?)\]\]"),
        "string_literal": re.compile(r'["\'](.*?)["\']'),
        "korean": re.compile(r"[가-힣]"),
    }
    
    def __init__(self, original_path: Path, translated_path: Path):
        self.original_path = original_path
        self.translated_path = translated_path
        self.issues = []

        try:
            self.original_lines = self.original_path.read_text('utf-8').splitlines()
            self.translated_lines = self.translated_path.read_text('utf-8').splitlines()
        except FileNotFoundError as e:
            print(f"오류: 파일을 찾을 수 없습니다 - {e}")
            exit(1)

    def _add_issue(self, **kwargs):
        """검증된 문제를 리스트에 추가합니다."""
        self.issues.append(kwargs)

    def run_validation_pipeline(self):
        """모든 검증 단계를 순차적으로 실행합니다."""
        print("1단계: 구조적 무결성 검사를 시작합니다...")
        is_structure_ok = self._check_line_count_and_structure()

        print("\n2.1단계: 코드 오염 검사를 시작합니다...")
        self._check_global_variable_consistency()
        self._check_macro_corruption()
        
        # 3번 요구사항 반영: 전역 검사와 별개로 라인별 변수 검사를 수행하여 모든 오류 위치를 기록
        if is_structure_ok:
            self._check_line_by_line_variable_consistency()
        else:
            print("\n경고: 파일 구조가 달라 라인별 변수 검사를 건너뜁니다. 구조를 먼저 수정하세요.")

        print(f"\n모든 검증 완료. 총 {len(self.issues)}개의 문제 발견.")

    def _check_line_count_and_structure(self) -> bool:
        """파일의 총 줄 수와 구조적 차이를 상세히 검사합니다."""
        if len(self.original_lines) == len(self.translated_lines):
            return True

        self._add_issue(
            line_num=0, severity="CRITICAL", type="구조적 오류",
            description=f"파일의 전체 줄 수가 일치하지 않습니다. (원본: {len(self.original_lines)}줄, 번역본: {len(self.translated_lines)}줄)"
        )

        matcher = difflib.SequenceMatcher(None, self.original_lines, self.translated_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            
            if tag == 'delete' or tag == 'insert':
                description = f"원본 {i1+1}번째 줄 근처에서 줄이 삭제/추가되었습니다. 원본과 구조를 동일하게 맞춰야 합니다."
                diff_lines = []
                # Context (Before)
                start = max(0, i1 - 2)
                for i in range(start, i1): diff_lines.append(f"  {self.original_lines[i]}")
                # Deleted lines
                for i in range(i1, i2): diff_lines.append(f"- {self.original_lines[i]}")
                # Inserted lines
                for j in range(j1, j2): diff_lines.append(f"+ {self.translated_lines[j]}")
                # Context (After)
                end = min(len(self.original_lines), i2 + 2)
                for i in range(i2, end): diff_lines.append(f"  {self.original_lines[i]}")

                self._add_issue(
                    line_num=i1 + 1, severity="CRITICAL", type="구조적 불일치 (줄 삭제/추가)",
                    description=description, diff_text="\n".join(diff_lines)
                )
        return False

    def _check_global_variable_consistency(self):
        """[전역 검사] 파일 전체의 변수 목록이 일치하는지 확인합니다."""
        original_vars = set(self.REGEX["variable"].findall("\n".join(self.original_lines)))
        translated_vars = set(self.REGEX["variable"].findall("\n".join(self.translated_lines)))

        missing = original_vars - translated_vars
        added = translated_vars - original_vars

        if missing:
            self._add_issue(
                severity="CRITICAL", type="[전역] 변수 누락",
                description=f"다음 변수들이 번역본 전체에서 누락되었습니다: {', '.join(sorted(list(missing)))}",
                line_num=0
            )
        if added:
            self._add_issue(
                severity="CRITICAL", type="[전역] 변수 손상/추가",
                description=f"다음 변수들이 잘못 추가/수정되었습니다: {', '.join(sorted(list(added)))}",
                line_num=0
            )

    def _check_line_by_line_variable_consistency(self):
        """[라인별 검사] 각 라인의 변수 목록이 원본과 일치하는지 확인합니다."""
        for i, (original_line, translated_line) in enumerate(zip(self.original_lines, self.translated_lines)):
            orig_vars = set(self.REGEX["variable"].findall(original_line))
            trans_vars = set(self.REGEX["variable"].findall(translated_line))

            if orig_vars != trans_vars:
                missing = orig_vars - trans_vars
                added = trans_vars - orig_vars
                desc = "해당 라인의 변수 목록이 원본과 다릅니다."
                if missing: desc += f" (누락: {', '.join(missing)})"
                if added: desc += f" (추가/손상: {', '.join(added)})"
                
                self._add_issue(
                    severity="CRITICAL", type="[라인별] 변수 불일치",
                    description=desc, line_num=i + 1,
                    original=original_line, translated=translated_line
                )

    def _check_macro_corruption(self):
        """매크로 내부에 한국어 문자열 리터럴이 있는지 검사합니다."""
        for i, translated_line in enumerate(self.translated_lines):
            translated_macros = self.REGEX["macro"].findall(translated_line)
            for trans_macro in translated_macros:
                literals = self.REGEX["string_literal"].findall(trans_macro)
                for literal in literals:
                    # 4번 요구사항 반영: 매크로 내부에 어떤 형태의 한국어든 발견되면 오류로 처리
                    if self.REGEX["korean"].search(literal):
                        self._add_issue(
                            severity="CRITICAL", type="매크로 코드 손상",
                            description=f"매크로 내부 코드에 한국어('{literal}')가 포함되었습니다. 이는 게임 오류를 유발할 수 있습니다. EasyPost 형식(`<<..._nun>>`)을 사용해야 합니다.",
                            line_num=i + 1,
                            original=f"원본 라인: `{self.original_lines[i]}`" if i < len(self.original_lines) else "원본 라인 없음",
                            translated=f"번역 라인: `{translated_line}`"
                        )
                        break # 한 매크로에서 여러 오류가 있어도 하나만 보고

    def generate_report(self, output_path: Path):
        """검증 결과를 Markdown 파일로 생성합니다."""
        report_content = f"# ❗ 종합 검증 리포트: {self.translated_path.name}\n\n"
        if not self.issues:
            report_content += "## ✅ 검증 완료: 발견된 문제가 없습니다."
        else:
            severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            sorted_issues = sorted(self.issues, key=lambda x: (x["line_num"], severity_order.get(x["severity"], 99)))
            
            summary = defaultdict(int)
            for issue in self.issues: summary[issue["severity"]] += 1

            report_content += f"## 요약\n\n- **총 문제 수: {len(self.issues)}**\n"
            if summary["CRITICAL"] > 0: report_content += f"- 🔴 **치명적 오류 (CRITICAL): {summary['CRITICAL']}**\n"
            
            report_content += "\n---\n\n## 상세 내용\n\n"

            for issue in sorted_issues:
                severity_icon = "🔴"
                report_content += f"### {severity_icon} [{issue['severity']}] {issue['type']} (원본 기준 Line: {issue['line_num'] or '전역'})\n\n"
                report_content += f"- **문제 설명:** {issue['description']}\n"
                if issue.get('diff_text'):
                    report_content += f"**차이점 분석 (Diff):**\n```diff\n{issue['diff_text']}\n```\n"
                if issue.get('original'):
                    report_content += f"- **원본:** `{issue['original']}`\n"
                if issue.get('translated'):
                    report_content += f"- **번역본:** `{issue['translated']}`\n"
                report_content += "\n---\n"
        
        output_path.write_text(report_content, 'utf-8')
        print(f"리포트가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # --- 설정: 여기에 검증할 파일 경로를 직접 입력하세요. ---
    ORIGINAL_FILE_PATH =
    TRANSLATED_FILE_PATH =
    OUTPUT_REPORT_PATH =
    # ----------------------------------------------------

    original_p = Path(ORIGINAL_FILE_PATH)
    translated_p = Path(TRANSLATED_FILE_PATH)
    output_p = Path(OUTPUT_REPORT_PATH)

    validator = TweeL10nValidator(original_path=original_p, translated_path=translated_p)
    validator.run_validation_pipeline()
    validator.generate_report(output_path=output_p)
