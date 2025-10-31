# Run Tests in Docker Environment (PowerShell)
# Comprehensive test execution script for Windows
#
# Author: Adrian Johnson <adrian207@gmail.com>

param(
    [Parameter(Position=0)]
    [ValidateSet('all', 'unit', 'integration', 'security', 'performance', 'smoke', 'coverage', 'clean', 'shell')]
    [string]$TestType = 'all',
    
    [Parameter(Position=1)]
    [switch]$Verbose
)

# Colors
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Green "🐳 Dual RAG LLM - Docker Test Environment"
Write-Output "==========================================="
Write-Output ""

# Function to run tests
function Run-Tests {
    param(
        [string]$TestPath,
        [string]$TestName,
        [string]$ExtraArgs = ""
    )
    
    Write-ColorOutput Blue "📋 Running $TestName..."
    
    docker-compose -f docker-compose.test.yml run --rm test-runner `
        pytest $TestPath -v $ExtraArgs `
        --cov=rag `
        --cov-report=html `
        --cov-report=term-missing
    
    Write-Output ""
}

# Start services
Write-ColorOutput Yellow "🚀 Starting test services..."
docker-compose -f docker-compose.test.yml up -d postgres redis

# Wait for services
Write-ColorOutput Yellow "⏳ Waiting for services to be ready..."
Start-Sleep -Seconds 5

# Check PostgreSQL
Write-Host "  PostgreSQL: " -NoNewline
try {
    $null = docker-compose -f docker-compose.test.yml exec -T postgres pg_isready -U testuser 2>&1
    Write-ColorOutput Green "✓"
} catch {
    Write-ColorOutput Red "✗"
    exit 1
}

# Check Redis
Write-Host "  Redis: " -NoNewline
try {
    $null = docker-compose -f docker-compose.test.yml exec -T redis redis-cli ping 2>&1
    Write-ColorOutput Green "✓"
} catch {
    Write-ColorOutput Red "✗"
    exit 1
}

Write-Output ""

# Run tests based on type
switch ($TestType) {
    'all' {
        Write-ColorOutput Green "🧪 Running ALL tests"
        Write-Output ""
        Run-Tests "tests/" "All Tests" "--maxfail=10"
    }
    
    'unit' {
        Run-Tests "tests/unit/" "Unit Tests" ""
    }
    
    'integration' {
        Run-Tests "tests/integration/" "Integration Tests" ""
    }
    
    'security' {
        Run-Tests "tests/security/" "Security Tests" "-m security"
    }
    
    'performance' {
        Run-Tests "tests/performance/" "Performance Tests" "-m performance --benchmark-only"
    }
    
    'smoke' {
        Write-ColorOutput Blue "💨 Running smoke tests (quick validation)"
        docker-compose -f docker-compose.test.yml run --rm test-runner `
            pytest tests/ -v -m smoke --maxfail=3
    }
    
    'coverage' {
        Write-ColorOutput Blue "📊 Running tests with detailed coverage"
        docker-compose -f docker-compose.test.yml run --rm test-runner `
            pytest tests/ -v `
            --cov=rag `
            --cov-report=html `
            --cov-report=term-missing `
            --cov-report=xml `
            --cov-fail-under=80
        
        Write-Output ""
        Write-ColorOutput Green "📈 Coverage report generated:"
        Write-Output "  HTML: htmlcov\index.html"
        Write-Output "  XML: coverage.xml"
    }
    
    'clean' {
        Write-ColorOutput Yellow "🧹 Cleaning up test environment..."
        docker-compose -f docker-compose.test.yml down -v
        Write-ColorOutput Green "✓ Cleanup complete"
        exit 0
    }
    
    'shell' {
        Write-ColorOutput Blue "🐚 Opening shell in test environment..."
        docker-compose -f docker-compose.test.yml run --rm test-runner bash
        exit 0
    }
}

# Get test results
$TestExitCode = $LASTEXITCODE

Write-Output ""
if ($TestExitCode -eq 0) {
    Write-ColorOutput Green "✅ Tests completed successfully!"
} else {
    Write-ColorOutput Red "❌ Tests failed with exit code $TestExitCode"
}

# Show info
Write-Output ""
Write-ColorOutput Blue "📊 Coverage report:"
Write-Output "  docker-compose -f docker-compose.test.yml run --rm test-runner cat htmlcov/index.html"
Write-Output ""
Write-ColorOutput Blue "🔍 View logs:"
Write-Output "  docker-compose -f docker-compose.test.yml logs"
Write-Output ""
Write-ColorOutput Blue "🧹 Cleanup:"
Write-Output "  docker-compose -f docker-compose.test.yml down -v"

exit $TestExitCode

