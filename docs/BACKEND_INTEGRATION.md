# 后端调用推荐API文档

## 背景

如果你的后端服务（Java/Go/PHP等）需要调用推荐API，并将推荐结果传递给前端，请参考本文档。

**关键点**：
- 推荐API现在会返回 `request_id` 字段
- 后端需要**透传** `request_id` 给前端
- 前端需要使用 `request_id` 进行埋点追踪

---

## API调用说明

### 1. 推荐API端点

#### 1.1 详情页推荐

```http
GET /recommend/detail/{dataset_id}?user_id={user_id}&limit=10
```

**路径参数**:
- `dataset_id` (int, 必需): 当前详情页的数据集ID

**查询参数**:
- `user_id` (int, 可选): 用户ID，用于个性化推荐
- `limit` (int, 可选): 返回推荐数量，默认20，范围1-100

**返回示例**:
```json
{
  "dataset_id": 1,
  "recommendations": [
    {
      "dataset_id": 42,
      "title": "某某数据集",
      "score": 0.95,
      "reason": "collaborative_filtering",
      "price": 199.0,
      "cover_image": "https://cdn.example.com/images/42.jpg"
    },
    {
      "dataset_id": 88,
      "title": "另一个数据集",
      "score": 0.87,
      "reason": "content_similarity",
      "price": 299.0,
      "cover_image": "https://cdn.example.com/images/88.jpg"
    }
  ],
  "request_id": "req_20251018_120530_abc123",  // 新增：唯一请求ID
  "algorithm_version": "20251018T120530Z"      // 新增：算法版本
}
```

#### 1.2 相似推荐

```http
GET /similar/{dataset_id}?limit=10
```

**路径参数**:
- `dataset_id` (int, 必需): 基准数据集ID

**查询参数**:
- `limit` (int, 可选): 返回推荐数量，默认20，范围1-100

**返回示例**:
```json
{
  "dataset_id": 1,
  "similar_items": [
    {
      "dataset_id": 42,
      "title": "相似数据集A",
      "score": 0.92,
      "reason": "content_similarity",
      "price": 199.0,
      "cover_image": "https://cdn.example.com/images/42.jpg"
    }
  ],
  "request_id": "req_20251018_120545_def456",
  "algorithm_version": "20251018T120530Z"
}
```

---

## 后端集成方式

### 场景1: 直接转发给前端（推荐）

如果后端只是透传推荐结果，**最简单的方式**是直接返回推荐API的完整响应：

#### Java (Spring Boot) 示例

```java
@RestController
@RequestMapping("/api/recommendations")
public class RecommendationController {

    @Autowired
    private RestTemplate restTemplate;

    @Value("${recommendation.api.url}")
    private String recommendApiUrl;  // http://recommendation-api:8000

    @GetMapping("/detail/{datasetId}")
    public ResponseEntity<Map<String, Object>> getRecommendations(
            @PathVariable Integer datasetId,
            @RequestParam(required = false) Integer userId,
            @RequestParam(defaultValue = "10") Integer limit) {

        // 调用推荐API
        String url = String.format(
            "%s/recommend/detail/%d?user_id=%d&limit=%d",
            recommendApiUrl, datasetId, userId, limit
        );

        // 直接返回完整响应（包含request_id）
        Map<String, Object> response = restTemplate.getForObject(
            url,
            Map.class
        );

        return ResponseEntity.ok(response);
    }
}
```

**配置文件** (`application.yml`):
```yaml
recommendation:
  api:
    url: http://recommendation-api:8000
```

#### Go (Gin) 示例

```go
package main

import (
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"

    "github.com/gin-gonic/gin"
)

const recommendAPIURL = "http://recommendation-api:8000"

type RecommendationResponse struct {
    DatasetID        int                  `json:"dataset_id"`
    Recommendations  []RecommendationItem `json:"recommendations"`
    RequestID        string               `json:"request_id"`
    AlgorithmVersion string               `json:"algorithm_version"`
}

type RecommendationItem struct {
    DatasetID  int     `json:"dataset_id"`
    Title      string  `json:"title"`
    Score      float64 `json:"score"`
    Reason     string  `json:"reason"`
    Price      float64 `json:"price"`
    CoverImage string  `json:"cover_image"`
}

func getRecommendations(c *gin.Context) {
    datasetID := c.Param("datasetId")
    userID := c.DefaultQuery("user_id", "")
    limit := c.DefaultQuery("limit", "10")

    // 调用推荐API
    url := fmt.Sprintf(
        "%s/recommend/detail/%s?user_id=%s&limit=%s",
        recommendAPIURL, datasetID, userID, limit,
    )

    resp, err := http.Get(url)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }
    defer resp.Body.Close()

    body, _ := ioutil.ReadAll(resp.Body)

    var result RecommendationResponse
    json.Unmarshal(body, &result)

    // 直接返回完整响应（包含request_id）
    c.JSON(http.StatusOK, result)
}

func main() {
    r := gin.Default()
    r.GET("/api/recommendations/detail/:datasetId", getRecommendations)
    r.Run(":8080")
}
```

#### PHP (Laravel) 示例

```php
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;

class RecommendationController extends Controller
{
    private $recommendApiUrl = 'http://recommendation-api:8000';

    public function getRecommendations(Request $request, $datasetId)
    {
        $userId = $request->query('user_id');
        $limit = $request->query('limit', 10);

        // 调用推荐API
        $url = "{$this->recommendApiUrl}/recommend/detail/{$datasetId}";
        $response = Http::get($url, [
            'user_id' => $userId,
            'limit' => $limit,
        ]);

        // 直接返回完整响应（包含request_id）
        return response()->json($response->json());
    }
}
```

**路由** (`routes/api.php`):
```php
Route::get('/recommendations/detail/{datasetId}', [RecommendationController::class, 'getRecommendations']);
```

#### Python (Django) 示例

```python
import requests
from django.http import JsonResponse
from django.conf import settings

RECOMMEND_API_URL = getattr(settings, 'RECOMMEND_API_URL', 'http://recommendation-api:8000')

def get_recommendations(request, dataset_id):
    user_id = request.GET.get('user_id')
    limit = request.GET.get('limit', 10)

    # 调用推荐API
    url = f"{RECOMMEND_API_URL}/recommend/detail/{dataset_id}"
    params = {'user_id': user_id, 'limit': limit}

    response = requests.get(url, params=params)

    # 直接返回完整响应（包含request_id）
    return JsonResponse(response.json())
```

---

### 场景2: 需要处理推荐数据

如果后端需要对推荐结果进行处理（如过滤、排序、增加字段等），**务必保留** `request_id` 并传给前端。

#### Java 示例

```java
@GetMapping("/detail/{datasetId}")
public ResponseEntity<CustomRecommendationResponse> getRecommendations(
        @PathVariable Integer datasetId,
        @RequestParam(required = false) Integer userId) {

    // 1. 调用推荐API
    String url = String.format("%s/recommend/detail/%d?user_id=%d&limit=20",
        recommendApiUrl, datasetId, userId);

    Map<String, Object> apiResponse = restTemplate.getForObject(url, Map.class);

    // 2. 提取推荐结果和request_id
    List<Map<String, Object>> recommendations =
        (List<Map<String, Object>>) apiResponse.get("recommendations");
    String requestId = (String) apiResponse.get("request_id");  // 关键！保留request_id

    // 3. 处理推荐数据（例如：过滤、增加字段）
    List<EnrichedRecommendationItem> enrichedItems = recommendations.stream()
        .limit(10)  // 只取前10个
        .map(item -> enrichItem(item))  // 自定义处理
        .collect(Collectors.toList());

    // 4. 返回时务必包含request_id
    CustomRecommendationResponse response = new CustomRecommendationResponse();
    response.setDatasetId(datasetId);
    response.setRecommendations(enrichedItems);
    response.setRequestId(requestId);  // 传给前端！

    return ResponseEntity.ok(response);
}

private EnrichedRecommendationItem enrichItem(Map<String, Object> item) {
    // 自定义处理逻辑，例如：
    // - 从数据库查询额外字段
    // - 计算折扣价格
    // - 过滤敏感信息
    EnrichedRecommendationItem enriched = new EnrichedRecommendationItem();
    enriched.setDatasetId((Integer) item.get("dataset_id"));
    enriched.setTitle((String) item.get("title"));
    enriched.setScore((Double) item.get("score"));
    // ... 其他字段
    return enriched;
}
```

**响应DTO**:
```java
public class CustomRecommendationResponse {
    private Integer datasetId;
    private List<EnrichedRecommendationItem> recommendations;
    private String requestId;  // 必须包含！
    private String algorithmVersion;

    // Getters and Setters
}
```

---

### 场景3: 后端聚合多个推荐源

如果后端需要聚合多个推荐源（如推荐API + 编辑推荐 + 热门榜单），需要为每个来源保留独立的 `request_id`。

#### Java 示例

```java
@GetMapping("/mixed/{datasetId}")
public ResponseEntity<MixedRecommendationResponse> getMixedRecommendations(
        @PathVariable Integer datasetId,
        @RequestParam(required = false) Integer userId) {

    MixedRecommendationResponse response = new MixedRecommendationResponse();

    // 1. 调用算法推荐API
    Map<String, Object> algoRec = callRecommendationAPI(datasetId, userId);
    response.setAlgorithmRecommendations(
        (List<Map<String, Object>>) algoRec.get("recommendations")
    );
    response.setAlgorithmRequestId((String) algoRec.get("request_id"));  // 保留request_id

    // 2. 获取编辑推荐
    List<Map<String, Object>> editorRec = getEditorRecommendations(datasetId);
    response.setEditorRecommendations(editorRec);

    // 3. 获取热门榜单
    List<Map<String, Object>> hotItems = getHotItems();
    response.setHotItems(hotItems);

    return ResponseEntity.ok(response);
}
```

**前端需要分别追踪**：
```javascript
// 算法推荐的点击
function trackAlgoClick(item, index, algorithmRequestId) {
  _paq.push(['setCustomDimension', 1, algorithmRequestId]);
  _paq.push(['trackEvent', 'Recommendation', 'AlgoClick', `dataset_${item.dataset_id}`, index]);
}

// 编辑推荐的点击
function trackEditorClick(item, index) {
  _paq.push(['trackEvent', 'Recommendation', 'EditorClick', `dataset_${item.dataset_id}`, index]);
}
```

---

## 错误处理

### 推荐API异常处理

```java
@GetMapping("/detail/{datasetId}")
public ResponseEntity<?> getRecommendations(
        @PathVariable Integer datasetId,
        @RequestParam(required = false) Integer userId) {

    try {
        String url = String.format("%s/recommend/detail/%d?user_id=%d",
            recommendApiUrl, datasetId, userId);

        Map<String, Object> response = restTemplate.getForObject(url, Map.class);
        return ResponseEntity.ok(response);

    } catch (HttpClientErrorException e) {
        // 推荐API返回4xx错误（如404）
        logger.warn("推荐API返回错误: {}", e.getMessage());
        return ResponseEntity.status(e.getStatusCode())
            .body(Map.of("error", "推荐服务暂时不可用"));

    } catch (HttpServerErrorException e) {
        // 推荐API返回5xx错误
        logger.error("推荐API服务器错误: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .body(Map.of("error", "推荐服务内部错误"));

    } catch (ResourceAccessException e) {
        // 推荐API网络不可达
        logger.error("无法连接推荐API: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
            .body(Map.of("error", "推荐服务不可用"));
    }
}
```

### 降级策略

当推荐API不可用时，可以返回默认推荐：

```java
private Map<String, Object> getFallbackRecommendations(Integer datasetId) {
    // 1. 从数据库查询同类别的数据集
    List<DatasetItem> fallbackItems = datasetRepository
        .findByCategoryAndNotId(getCategoryByDatasetId(datasetId), datasetId)
        .limit(10)
        .collect(Collectors.toList());

    // 2. 构造响应（注意：没有真实的request_id）
    Map<String, Object> response = new HashMap<>();
    response.put("dataset_id", datasetId);
    response.put("recommendations", fallbackItems);
    response.put("request_id", "fallback_" + System.currentTimeMillis());  // 降级标记
    response.put("algorithm_version", "fallback");

    return response;
}
```

---

## 配置建议

### 1. 超时设置

推荐API响应时间通常在200ms以内，建议设置超时：

```java
// Spring Boot
@Bean
public RestTemplate restTemplate() {
    HttpComponentsClientHttpRequestFactory factory =
        new HttpComponentsClientHttpRequestFactory();
    factory.setConnectTimeout(2000);  // 连接超时2秒
    factory.setReadTimeout(5000);     // 读取超时5秒
    return new RestTemplate(factory);
}
```

### 2. 连接池

生产环境建议使用连接池：

```java
@Bean
public RestTemplate restTemplate() {
    PoolingHttpClientConnectionManager cm =
        new PoolingHttpClientConnectionManager();
    cm.setMaxTotal(100);  // 最大连接数
    cm.setDefaultMaxPerRoute(20);  // 每个路由最大连接数

    CloseableHttpClient httpClient = HttpClients.custom()
        .setConnectionManager(cm)
        .build();

    HttpComponentsClientHttpRequestFactory factory =
        new HttpComponentsClientHttpRequestFactory(httpClient);

    return new RestTemplate(factory);
}
```

### 3. 缓存（可选）

对于相同的请求，可以短暂缓存（30秒-1分钟）：

```java
@Cacheable(value = "recommendations", key = "#datasetId + '_' + #userId", unless = "#result == null")
public Map<String, Object> getRecommendations(Integer datasetId, Integer userId) {
    // 调用推荐API...
}
```

**注意**：缓存会导致多个请求共享同一个 `request_id`，影响追踪准确性。建议：
- 缓存时间不超过1分钟
- 或者只缓存推荐列表，不缓存 `request_id`（每次生成新的）

---

## 测试验证

### 1. 单元测试

```java
@SpringBootTest
@AutoConfigureMockMvc
public class RecommendationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private RestTemplate restTemplate;

    @Test
    public void testGetRecommendations() throws Exception {
        // Mock推荐API响应
        Map<String, Object> mockResponse = new HashMap<>();
        mockResponse.put("dataset_id", 1);
        mockResponse.put("recommendations", List.of());
        mockResponse.put("request_id", "test_request_id");
        mockResponse.put("algorithm_version", "v1.0.0");

        when(restTemplate.getForObject(anyString(), eq(Map.class)))
            .thenReturn(mockResponse);

        // 调用接口
        mockMvc.perform(get("/api/recommendations/detail/1?user_id=123"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.request_id").value("test_request_id"))
            .andExpect(jsonPath("$.dataset_id").value(1));
    }
}
```

### 2. 集成测试

```bash
# 调用后端接口
curl http://localhost:8080/api/recommendations/detail/1?user_id=123

# 验证返回包含request_id
{
  "dataset_id": 1,
  "recommendations": [...],
  "request_id": "req_20251018_120530_abc123",  // 必须有这个字段
  "algorithm_version": "20251018T120530Z"
}
```

### 3. 监控指标

建议添加监控指标：

```java
@Autowired
private MeterRegistry meterRegistry;

public Map<String, Object> getRecommendations(Integer datasetId, Integer userId) {
    Timer.Sample sample = Timer.start(meterRegistry);

    try {
        Map<String, Object> response = callRecommendationAPI(datasetId, userId);

        // 记录成功
        meterRegistry.counter("recommendation.api.success").increment();

        return response;

    } catch (Exception e) {
        // 记录失败
        meterRegistry.counter("recommendation.api.failure").increment();
        throw e;

    } finally {
        sample.stop(Timer.builder("recommendation.api.latency")
            .tag("endpoint", "detail")
            .register(meterRegistry));
    }
}
```

---

## 验证清单

部署前检查：

- [ ] 调用推荐API能正常返回 `request_id` 字段
- [ ] 后端接口透传了 `request_id` 给前端
- [ ] 错误处理和降级策略已实现
- [ ] 设置了合理的超时时间
- [ ] 添加了监控指标
- [ ] 单元测试覆盖主要场景

部署后验证：

- [ ] 前端能正常接收到 `request_id`
- [ ] 查看日志确认推荐API调用成功
- [ ] 监控面板显示推荐API延迟正常
- [ ] 前端埋点能正常发送（通知前端团队验证）

---

## 常见问题

### Q1: request_id 格式是什么？

**A**: 格式为 `req_{timestamp}_{random}`，例如 `req_20251018_120530_abc123`。这是唯一的，每次API调用都会生成新的。

### Q2: 是否需要在后端存储 request_id？

**A**: **不需要**。`request_id` 仅用于关联前端埋点和后端曝光日志，由推荐系统负责存储和分析。

### Q3: 如果后端修改了推荐结果（如过滤、排序），是否需要新的 request_id？

**A**: **不需要**。保留原始的 `request_id`。即使你修改了推荐列表，后端的评估逻辑会基于**用户实际看到的列表**计算CTR。

### Q4: 降级时的 request_id 怎么处理？

**A**: 降级时可以生成特殊的 `request_id`，例如 `fallback_{timestamp}`，方便后续分析降级场景的效果。

---

## 联系方式

如有疑问，请联系推荐系统团队：
- 负责人：[你的名字]
- 邮箱：[你的邮箱]
- 推荐API文档：http://recommendation-api:8000/docs
- 文档更新日期：2025-10-18
