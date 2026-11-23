import SwiftUI

@main
struct LittleBitLoaderApp: App {
    @State private var selectedTab: Int = 0

    var body: some Scene {
        WindowGroup {
            TabView(selection: $selectedTab) {
                HomeView(selectedTab: $selectedTab)
                    .tabItem {
                        Label("الرئيسية", systemImage: "house.fill")
                    }
                    .tag(0)

                ModelManagerView()
                    .tabItem {
                        Label("النماذج", systemImage: "folder.fill")
                    }
                    .tag(1)

                ChatView()
                    .tabItem {
                        Label("المحادثة", systemImage: "message.fill")
                    }
                    .tag(2)

                PerformanceMonitorView()
                    .tabItem {
                        Label("الأداء", systemImage: "speedometer")
                    }
                    .tag(3)
            }
        }
    }
}
