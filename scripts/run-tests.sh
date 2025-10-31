#!/bin/bash
#
# Run Tests in Docker Environment
# Comprehensive test execution script
#
# Author: Adrian Johnson <adrian207@gmail.com>

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐳 Dual RAG LLM - Docker Test Environment${NC}"
echo "==========================================="
echo ""

# Parse command line arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:-false}"

# Function to run specific test type
run_tests() {
    local test_path=$1
    local test_name=$2
    local extra_args=$3
    
    echo -e "${BLUE}📋 Running ${test_name}...${NC}"
    
    docker-compose -f docker-compose.test.yml run --rm test-runner \
        pytest ${test_path} -v ${extra_args} \
        --cov=rag \
        --cov-report=html \
        --cov-report=term-missing
    
    echo ""
}

# Start services
echo -e "${YELLOW}🚀 Starting test services...${NC}"
docker-compose -f docker-compose.test.yml up -d postgres redis

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 5

# Check PostgreSQL
echo -n "  PostgreSQL: "
if docker-compose -f docker-compose.test.yml exec -T postgres pg_isready -U testuser > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

# Check Redis
echo -n "  Redis: "
if docker-compose -f docker-compose.test.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    exit 1
fi

echo ""

# Run tests based on type
case ${TEST_TYPE} in
    all)
        echo -e "${GREEN}🧪 Running ALL tests${NC}"
        echo ""
        run_tests "tests/" "All Tests" "--maxfail=10"
        ;;
    
    unit)
        run_tests "tests/unit/" "Unit Tests" ""
        ;;
    
    integration)
        run_tests "tests/integration/" "Integration Tests" ""
        ;;
    
    security)
        run_tests "tests/security/" "Security Tests" "-m security"
        ;;
    
    performance)
        run_tests "tests/performance/" "Performance Tests" "-m performance --benchmark-only"
        ;;
    
    smoke)
        echo -e "${BLUE}💨 Running smoke tests (quick validation)${NC}"
        docker-compose -f docker-compose.test.yml run --rm test-runner \
            pytest tests/ -v -m smoke --maxfail=3
        ;;
    
    coverage)
        echo -e "${BLUE}📊 Running tests with detailed coverage${NC}"
        docker-compose -f docker-compose.test.yml run --rm test-runner \
            pytest tests/ -v \
            --cov=rag \
            --cov-report=html \
            --cov-report=term-missing \
            --cov-report=xml \
            --cov-fail-under=80
        
        echo ""
        echo -e "${GREEN}📈 Coverage report generated:${NC}"
        echo "  HTML: htmlcov/index.html"
        echo "  XML: coverage.xml"
        ;;
    
    clean)
        echo -e "${YELLOW}🧹 Cleaning up test environment...${NC}"
        docker-compose -f docker-compose.test.yml down -v
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        exit 0
        ;;
    
    shell)
        echo -e "${BLUE}🐚 Opening shell in test environment...${NC}"
        docker-compose -f docker-compose.test.yml run --rm test-runner bash
        exit 0
        ;;
    
    *)
        echo -e "${RED}Unknown test type: ${TEST_TYPE}${NC}"
        echo ""
        echo "Usage: $0 [test-type] [verbose]"
        echo ""
        echo "Test Types:"
        echo "  all          - Run all tests (default)"
        echo "  unit         - Run unit tests only"
        echo "  integration  - Run integration tests only"
        echo "  security     - Run security tests only"
        echo "  performance  - Run performance benchmarks"
        echo "  smoke        - Run quick smoke tests"
        echo "  coverage     - Run with detailed coverage report"
        echo "  clean        - Clean up test environment"
        echo "  shell        - Open shell in test container"
        echo ""
        exit 1
        ;;
esac

# Get test results
TEST_EXIT_CODE=$?

echo ""
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}✅ Tests completed successfully!${NC}"
else
    echo -e "${RED}❌ Tests failed with exit code ${TEST_EXIT_CODE}${NC}"
fi

# Show coverage report location
echo ""
echo -e "${BLUE}📊 Coverage report:${NC}"
echo "  docker-compose -f docker-compose.test.yml run --rm test-runner cat htmlcov/index.html"
echo ""
echo -e "${BLUE}🔍 View logs:${NC}"
echo "  docker-compose -f docker-compose.test.yml logs"
echo ""
echo -e "${BLUE}🧹 Cleanup:${NC}"
echo "  docker-compose -f docker-compose.test.yml down -v"

exit ${TEST_EXIT_CODE}

