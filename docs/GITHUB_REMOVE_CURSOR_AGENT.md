# GitHub 기여자 목록에서 "Cursor Agent" 제거하기

이 저장소의 **커밋 기록**에는 모든 커밋이 **Heuiseung Jeong**으로만 되어 있습니다.  
그래도 기여자 목록에 "Cursor Agent" 또는 "cursoragent"가 보인다면, **협력자(Collaborator)**로 등록된 경우입니다.

## 방법: 협력자에서 제거

1. GitHub에서 해당 저장소로 이동  
   `https://github.com/heuiseung/VitalDB-Hypotension-Prediction`

2. **Settings** 탭 클릭

3. 왼쪽 메뉴에서 **Collaborators and teams** (또는 **Manage access**) 선택

4. 목록에서 **Cursor Agent**, **cursoragent** 또는 관련 계정 찾기

5. 해당 계정 오른쪽 **Remove** 또는 메뉴(⋮) → **Remove** 클릭 후 확인

이렇게 하면 해당 계정의 저장소 접근 권한이 없어지고, 기여자 목록에서도 사라질 수 있습니다. (캐시 때문에 바로 안 사라질 수 있음)

## 참고

- **Contributors** 탭은 Git 커밋의 **author** 정보로 자동 생성됩니다.
- 현재 이 저장소의 모든 커밋 author는 `Heuiseung Jeong <sck326@naver.com>` 뿐이므로, 커밋 기록 때문에 "Cursor Agent"가 기여자로 나오는 것은 아닙니다.
- Cursor에서 커밋할 때 사용하는 **Git 사용자 이름/이메일**이 "Cursor Agent"로 설정돼 있었다면, 앞으로는 `git config user.name "Heuiseung Jeong"`, `user.email "sck326@naver.com"`으로 설정해 두면 새 커밋은 본인 이름으로만 기록됩니다.
