// ==========================================
// Theme Toggle Functionality
// ==========================================
function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = document.querySelector('.theme-icon');
    const html = document.documentElement;

    // Check for saved theme preference or default to light mode
    const currentTheme = localStorage.getItem('theme') || 'light';
    html.setAttribute('data-theme', currentTheme);
    updateThemeIcon(currentTheme, themeIcon);

    // Theme toggle event listener
    themeToggle.addEventListener('click', () => {
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon(newTheme, themeIcon);
    });
}

function updateThemeIcon(theme, iconElement) {
    iconElement.textContent = theme === 'light' ? '🌙' : '☀️';
}

// ==========================================
// Project Filtering Functionality
// ==========================================
function initializeProjectFilters() {
    const filterButtons = document.querySelectorAll('.filter-btn');
    const projectCards = document.querySelectorAll('.project-card');

    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));

            // Add active class to clicked button
            button.classList.add('active');

            // Get filter value
            const filterValue = button.getAttribute('data-filter');

            // Filter projects
            projectCards.forEach(card => {
                if (filterValue === 'all') {
                    card.classList.remove('hidden');
                    animateCardIn(card);
                } else {
                    const categories = card.getAttribute('data-category');
                    if (categories.includes(filterValue)) {
                        card.classList.remove('hidden');
                        animateCardIn(card);
                    } else {
                        card.classList.add('hidden');
                    }
                }
            });
        });
    });
}

function animateCardIn(card) {
    card.style.animation = 'none';
    setTimeout(() => {
        card.style.animation = 'fadeIn 0.6s ease-out';
    }, 10);
}

// ==========================================
// Smooth Scroll with Offset for Fixed Header
// ==========================================
function initializeSmoothScroll() {
    const navLinks = document.querySelectorAll('.nav-link');
    const headerHeight = document.querySelector('.header').offsetHeight;

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);

            if (targetSection) {
                const targetPosition = targetSection.offsetTop - headerHeight - 20;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// ==========================================
// Scroll-based Animations
// ==========================================
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe elements for animation
    const animateElements = document.querySelectorAll(
        '.achievement-card, .project-card, .skill-category, .contact-card'
    );

    animateElements.forEach(element => {
        observer.observe(element);
    });
}

// ==========================================
// Header Scroll Effect
// ==========================================
function initializeHeaderScrollEffect() {
    const header = document.querySelector('.header');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            header.style.boxShadow = 'var(--shadow-lg)';
        } else {
            header.style.boxShadow = 'var(--shadow-sm)';
        }

        lastScroll = currentScroll;
    });
}

// ==========================================
// Stats Counter Animation
// ==========================================
function animateStatsCounter() {
    const statValues = document.querySelectorAll('.stat-value');

    const observerOptions = {
        threshold: 0.5
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
                animateValue(entry.target);
                entry.target.classList.add('counted');
            }
        });
    }, observerOptions);

    statValues.forEach(stat => observer.observe(stat));
}

function animateValue(element) {
    const text = element.textContent;

    // Skip animation for non-numeric values
    if (text.includes('K+') || text.includes('×') || text.includes('%')) {
        return;
    }

    const start = 0;
    const end = parseFloat(text);
    const duration = 1500;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const current = start + (end - start) * easeOutCubic(progress);

        if (text.includes('%')) {
            element.textContent = current.toFixed(2) + '%';
        } else {
            element.textContent = current.toFixed(text.includes('.') ? 2 : 0);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function easeOutCubic(x) {
    return 1 - Math.pow(1 - x, 3);
}

// ==========================================
// Project Card Hover Effects
// ==========================================
function initializeProjectCardEffects() {
    const projectCards = document.querySelectorAll('.project-card');

    projectCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.3s ease';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transition = 'all 0.25s ease-in-out';
        });
    });
}

// ==========================================
// Mobile Menu Toggle (for future enhancement)
// ==========================================
function initializeMobileMenu() {
    // Placeholder for mobile menu functionality
    // Can be expanded if hamburger menu is added
    const isMobile = window.innerWidth < 768;

    if (isMobile) {
        console.log('Mobile view detected');
    }
}

// ==========================================
// Active Navigation Link Highlighting
// ==========================================
function initializeActiveNavigation() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        const scrollPosition = window.pageYOffset;

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 150;
            const sectionHeight = section.offsetHeight;

            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ==========================================
// Lazy Loading Images (if images are added)
// ==========================================
function initializeLazyLoading() {
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.add('loaded');
                    imageObserver.unobserve(img);
                }
            });
        });

        const images = document.querySelectorAll('img[data-src]');
        images.forEach(img => imageObserver.observe(img));
    }
}

// ==========================================
// Performance Monitoring
// ==========================================
function logPerformanceMetrics() {
    if (window.performance && window.performance.timing) {
        window.addEventListener('load', () => {
            setTimeout(() => {
                const perfData = window.performance.timing;
                const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
                const connectTime = perfData.responseEnd - perfData.requestStart;
                const renderTime = perfData.domComplete - perfData.domLoading;

                console.log('Performance Metrics:');
                console.log(`Page Load Time: ${pageLoadTime}ms`);
                console.log(`Connect Time: ${connectTime}ms`);
                console.log(`Render Time: ${renderTime}ms`);
            }, 0);
        });
    }
}

// ==========================================
// Accessibility Enhancements
// ==========================================
function initializeAccessibility() {
    // Add keyboard navigation for custom elements
    const filterButtons = document.querySelectorAll('.filter-btn');

    filterButtons.forEach((button, index) => {
        button.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowRight') {
                e.preventDefault();
                const nextButton = filterButtons[index + 1] || filterButtons[0];
                nextButton.focus();
            } else if (e.key === 'ArrowLeft') {
                e.preventDefault();
                const prevButton = filterButtons[index - 1] || filterButtons[filterButtons.length - 1];
                prevButton.focus();
            } else if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                button.click();
            }
        });
    });

    // Ensure focus is visible
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-nav');
        }
    });

    document.addEventListener('mousedown', () => {
        document.body.classList.remove('keyboard-nav');
    });
}

// ==========================================
// Error Handling
// ==========================================
function initializeErrorHandling() {
    window.addEventListener('error', (e) => {
        console.error('An error occurred:', e.error);
        // Could add user-friendly error messages here
    });

    window.addEventListener('unhandledrejection', (e) => {
        console.error('Unhandled promise rejection:', e.reason);
    });
}

// ==========================================
// Copy to Clipboard Functionality (for future use)
// ==========================================
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            console.log('Copied to clipboard:', text);
        }).catch(err => {
            console.error('Failed to copy:', err);
        });
    }
}

// ==========================================
// Initialize All Functions on DOM Load
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing ML Research Engineer Portfolio...');

    try {
        // Core functionality
        initializeTheme();
        initializeProjectFilters();
        initializeSmoothScroll();
        initializeScrollAnimations();
        initializeHeaderScrollEffect();

        // Enhanced features
        animateStatsCounter();
        initializeProjectCardEffects();
        initializeMobileMenu();
        initializeActiveNavigation();
        initializeLazyLoading();
        initializeAccessibility();
        initializeErrorHandling();

        // Performance monitoring (dev mode)
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            logPerformanceMetrics();
        }

        console.log('Portfolio initialized successfully! ✅');
    } catch (error) {
        console.error('Error initializing portfolio:', error);
    }
});

// ==========================================
// Export functions for testing (if needed)
// ==========================================
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeTheme,
        initializeProjectFilters,
        animateValue,
        copyToClipboard
    };
}
