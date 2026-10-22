<template>
    <UDashboardGroup>
        <UDashboardSidebar resizable :default-size="20" :min-size="15" :max-size="25" class="border-r border-default" :ui="{ body: 'p-0 gap-0' }">
            <Sidebar />
        </UDashboardSidebar>

        <UDashboardPanel grow>
            <div class="h-screen w-full flex">
                <MainPanel />
            </div>
        </UDashboardPanel>

        <UDashboardSidebar side="right" resizable :default-size="25" :min-size="20" :max-size="50" class="border-l border-default" :ui="{ body: 'p-0 gap-0' }">
            <MapPanel class="px-0 py-0 gap-0" />
        </UDashboardSidebar>

    </UDashboardGroup>
</template>

