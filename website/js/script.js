// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Mobile Navigation Toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');

    if (navToggle) {
        navToggle.addEventListener('click', function() {
            navToggle.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Close mobile menu when clicking on a nav link
    const navLinks = document.querySelectorAll('.nav-menu a');
    navLinks.forEach(link => {
        link.addEventListener('click', function() {
            navToggle.classList.remove('active');
            navMenu.classList.remove('active');
        });
    });

    // Testimonial Slider
    const testimonials = document.querySelectorAll('.testimonial');
    const dots = document.querySelectorAll('.dot');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');

    if (testimonials.length > 0) {
        let currentSlide = 0;

        // Hide all testimonials except the first one
        testimonials.forEach((testimonial, index) => {
            if (index !== 0) {
                testimonial.style.display = 'none';
            }
        });

        // Function to show a specific slide
        function showSlide(n) {
            // Hide all testimonials
            testimonials.forEach(testimonial => {
                testimonial.style.display = 'none';
            });

            // Remove active class from all dots
            dots.forEach(dot => {
                dot.classList.remove('active');
            });

            // Show the selected testimonial
            testimonials[n].style.display = 'block';
            
            // Add active class to the corresponding dot
            dots[n].classList.add('active');
            
            // Update current slide
            currentSlide = n;
        }

        // Next button click event
        if (nextBtn) {
            nextBtn.addEventListener('click', function() {
                currentSlide = (currentSlide + 1) % testimonials.length;
                showSlide(currentSlide);
            });
        }

        // Previous button click event
        if (prevBtn) {
            prevBtn.addEventListener('click', function() {
                currentSlide = (currentSlide - 1 + testimonials.length) % testimonials.length;
                showSlide(currentSlide);
            });
        }

        // Dot click events
        dots.forEach((dot, index) => {
            dot.addEventListener('click', function() {
                showSlide(index);
            });
        });

        // Auto slide every 5 seconds
        setInterval(function() {
            currentSlide = (currentSlide + 1) % testimonials.length;
            showSlide(currentSlide);
        }, 5000);
    }

    // Demo Page Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    if (tabBtns.length > 0) {
        tabBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                // Remove active class from all buttons
                tabBtns.forEach(btn => {
                    btn.classList.remove('active');
                });

                // Add active class to clicked button
                this.classList.add('active');

                // Hide all tab panes
                tabPanes.forEach(pane => {
                    pane.classList.remove('active');
                });

                // Show the corresponding tab pane
                const tabId = this.getAttribute('data-tab');
                document.getElementById(`${tabId}-tab`).classList.add('active');
            });
        });
    }

    // Features Page Category Tabs
    const categoryTabs = document.querySelectorAll('.category-tab');
    const featureItems = document.querySelectorAll('.feature-item');

    if (categoryTabs.length > 0) {
        categoryTabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                categoryTabs.forEach(tab => {
                    tab.classList.remove('active');
                });

                // Add active class to clicked tab
                this.classList.add('active');

                // Get the selected category
                const category = this.getAttribute('data-category');

                // Show/hide feature items based on category
                featureItems.forEach(item => {
                    if (category === 'all') {
                        item.style.display = 'flex';
                    } else {
                        const itemCategories = item.getAttribute('data-categories').split(' ');
                        if (itemCategories.includes(category)) {
                            item.style.display = 'flex';
                        } else {
                            item.style.display = 'none';
                        }
                    }
                });
            });
        });
    }

    // Streamlit iframe loading animation
    const streamlitIframe = document.getElementById('streamlit-iframe');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.querySelector('.loading-status');

    if (streamlitIframe && loadingOverlay) {
        // Update loading status messages
        const loadingMessages = [
            'Initializing application...',
            'Loading data processing modules...',
            'Setting up trading strategies...',
            'Configuring risk analysis tools...',
            'Preparing visualization components...',
            'Almost ready...'
        ];

        let messageIndex = 0;
        const messageInterval = setInterval(function() {
            if (messageIndex < loadingMessages.length) {
                loadingStatus.textContent = loadingMessages[messageIndex];
                messageIndex++;
            } else {
                clearInterval(messageInterval);
            }
        }, 1500);

        // Hide loading overlay when iframe is loaded
        streamlitIframe.addEventListener('load', function() {
            // Give a little extra time for Streamlit to fully render
            setTimeout(function() {
                loadingOverlay.style.opacity = '0';
                setTimeout(function() {
                    loadingOverlay.style.display = 'none';
                }, 500);
            }, 2000);
        });

        // Fallback: Hide loading overlay after 20 seconds even if iframe doesn't trigger load event
        setTimeout(function() {
            if (loadingOverlay.style.display !== 'none') {
                loadingOverlay.style.opacity = '0';
                setTimeout(function() {
                    loadingOverlay.style.display = 'none';
                }, 500);
            }
        }, 20000);
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
            }
        });
    });
});
