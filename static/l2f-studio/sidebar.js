document.addEventListener('DOMContentLoaded', () => {
    const sidebarResizer = document.getElementById('sidebar-resizer');
    const vehicleContainer = document.getElementById('vehicle-container');
    const simContainer = document.getElementById('sim-container');
    let isResizing = false;
    let dragStartX = 0; 
    let didMove = false; 
    if(localStorage.getItem('vehicle-container-width') !== null) {
        vehicleContainer.style.width = localStorage.getItem('vehicle-container-width')
        window.dispatchEvent(new Event('resize'));
    }

    // Mouse and touch events
    sidebarResizer.addEventListener('mousedown', initResize);
    sidebarResizer.addEventListener('touchstart', initResize, { passive: false });
    document.addEventListener('mousemove', resize);
    document.addEventListener('touchmove', resize, { passive: false });
    document.addEventListener('mouseup', stopResize);
    document.addEventListener('touchend', stopResize);

    sidebarResizer.addEventListener('click', e => {
        if (didMove) {
            e.stopPropagation();
            e.preventDefault();
            return;
        }
        let newWidth = '0px';
        if(parseInt(vehicleContainer.style.width) < 10 || !vehicleContainer.style.width){
            newWidth = '200px';
        }
        vehicleContainer.style.width = newWidth;
        localStorage.setItem('vehicle-container-width', newWidth);
        window.dispatchEvent(new Event('resize'));
    });

    function getClientX(e) {
        return e.touches ? e.touches[0].clientX : e.clientX;
    }

    function initResize(e) {
        if (e.touches && e.touches.length !== 1) return;
        isResizing = true;
        didMove = false;
        dragStartX = getClientX(e);
        document.body.style.cursor = 'col-resize';
        e.preventDefault();
    }

    function resize(e) {
        if (!isResizing) return;
        if (e.touches && e.touches.length !== 1) return;

        const clientX = getClientX(e);
        if (Math.abs(clientX - dragStartX) > 3) {
            didMove = true;
        }
        
        const containerRect = sidebarResizer.parentElement.getBoundingClientRect();
        let newWidth = clientX - containerRect.left - sidebarResizer.getBoundingClientRect().width / 2;
        newWidth = Math.max(0, newWidth)

        
        vehicleContainer.style.width = `${newWidth}px`;
        simContainer.style.flexGrow = 1;
        window.dispatchEvent(new Event('resize'));
        localStorage.setItem('vehicle-container-width', newWidth);
        
        e.preventDefault();
    }

    function stopResize() {
        isResizing = false;
        document.body.style.cursor = 'default';
        window.dispatchEvent(new Event('resize'));
    }

});